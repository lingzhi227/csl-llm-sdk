"""Detached host supervisor adapted from the accepted physical smoke lifecycle."""
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import subprocess
import sys
import time
from .store import atomic_json

TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "CANCELED", "DELETED"}


def query(*args):
    p = subprocess.run(["csctl", "get", *args, "-o", "json"], capture_output=True,
                       text=True, check=True, timeout=25)
    return json.loads(p.stdout)


def ids(items):
    return {j["meta"]["name"].split("/")[-1] for j in items["items"]}


def captured_ids(root, name):
    result = set()
    errors = []
    for path in (root / (name + ".jobs"), root / (name + ".log")):
        try:
            if path.exists():
                limit = 65536 if path.suffix == ".jobs" else 8 << 20
                with path.open() as stream:
                    raw = stream.read(limit)
                    if stream.read(1):
                        errors.append(path.name + " exceeds bound; parsed bounded prefix")
                pattern = r"(?m)^(wsjob-[a-z0-9]+)$" if path.suffix == ".jobs" else r"Job id: (wsjob-[a-z0-9]+)\b"
                result.update(re.findall(pattern, raw))
        except BaseException as exc:
            errors.append(path.name + ": " + str(exc))
    return result, errors


def release(root, name, submitted, query_fn=query, cancel_fn=None, timeout=90):
    """An ID must be invocation-correlated AND owned before cancellation."""
    owner = getpass.getuser()
    records = []
    if cancel_fn is None:
        def cancel_fn(jid):
            return subprocess.run(["csctl", "cancel", "job", jid], capture_output=True,
                                  text=True, check=True, timeout=25)
    for jid in sorted(submitted):
        job = query_fn("job", jid)
        if job["spec"]["user"]["username"] != owner:
            raise RuntimeError("Refusing foreign job cancellation: " + jid)
        cancelled = job["status"]["phase"] not in TERMINAL
        if cancelled:
            cancel_fn(jid)
        deadline = time.monotonic() + timeout
        while True:
            job = query_fn("job", jid)
            systems = query_fn("systems")
            assigned = [s["meta"]["name"] for s in systems["items"]
                        if jid in {str(v).split("/")[-1] for v in [s.get("jobId", ""), *s.get("jobIds", [])]}]
            if job["status"]["phase"] in TERMINAL and not assigned:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Job release not confirmed: " + jid)
            time.sleep(2)
        atomic_json(root / (jid + ".json"), job)
        atomic_json(root / (name + "-systems-after.json"), systems)
        records.append(dict(id=jid, phase=job["status"]["phase"], released=True, cancelled=cancelled))
    return records


def child_limits():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 << 20, 128 << 20))


def terminate(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def cleanup(root, name, baseline, proc, query_fn=query, release_fn=release, terminate_fn=terminate):
    """Each cleanup branch runs even if an earlier branch failed."""
    errors, records = [], []
    if proc is not None:
        try:
            terminate_fn(proc)
        except BaseException as exc:
            errors.append("terminate: " + str(exc))
    submitted, capture_errors = captured_ids(root, name)
    errors.extend(capture_errors)
    old = submitted & ids(baseline)
    if old:
        errors.append("Refused pre-existing IDs: " + str(sorted(old)))
    proven = submitted - old
    for jid in sorted(proven):
        try:
            records.extend(release_fn(root, name, {jid}, query_fn=query_fn))
        except BaseException as exc:
            errors.append("release " + jid + ": " + str(exc))
    unknown = []
    for label, args in (("jobs", ("jobs", "--my-jobs", "--all-states")), ("systems", ("systems",))):
        try:
            after = query_fn(*args)
            atomic_json(root / (name + "-" + label + "-after.json"), after)
            if label == "jobs":
                unknown = sorted(ids(after) - ids(baseline) - proven)
        except BaseException as exc:
            errors.append("final " + label + ": " + str(exc))
    if unknown:
        errors.append("Uncorrelated new account jobs: " + str(unknown))
    if not proven:
        errors.append("No submitted job proven; cleanup is not claimed")
    return dict(jobs=records, cleanup_errors=errors, uncorrelated_new_jobs=unknown)


def stage(root, name, script, timeout):
    from source_gate import PROFILE
    from runtime_monitor import Monitor
    baseline = query("jobs", "--my-jobs", "--all-states")
    atomic_json(root / (name + "-jobs-before.json"), baseline)
    started = time.monotonic(); monitor = Monitor(root, PROFILE, started)
    proc = None; error = None; code = None
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               HW00_JOB_CAPTURE=str(root / (name + ".jobs")))
    try:
        with (root / (name + ".log")).open("x") as log:
            proc = subprocess.Popen([sys.executable, script], cwd=root, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=child_limits)
            atomic_json(root / "ACTIVE.json", dict(phase=name, pid=proc.pid, timeout=timeout, source="dense-transport"))
            while proc.poll() is None:
                monitor.check(time.monotonic())
                if shutil.disk_usage(root).free < PROFILE["host_disk_reserve_bytes"]:
                    raise RuntimeError("Host disk reserve crossed")
                sizes=[]
                for p in root.rglob("*"):
                    try:
                        if p.is_symlink():raise RuntimeError("Candidate symlink forbidden")
                        if p.is_file():sizes.append(p.stat().st_size)
                    except FileNotFoundError:
                        # Atomic record replacement can remove a listed temp.
                        # Frozen inputs are verified before and after the run.
                        continue
                    if len(sizes)>2048:raise RuntimeError("Candidate file count cap")
                if sum(sizes)>PROFILE["candidate_bytes"]:raise RuntimeError("Candidate storage budget exceeded")
                for filename in (name+'.log','supervisor.log'):
                    path=root/filename
                    if path.exists() and path.stat().st_size>PROFILE['host_log_bytes']:
                        raise RuntimeError("Host log budget exceeded")
                stack_log=root/'startup-stack.log'
                if stack_log.exists() and stack_log.stat().st_size>PROFILE['startup_stack_bytes']:
                    raise RuntimeError('Startup stack sampled byte cap')
                # RSS is sampled; RLIMIT_AS is the hard address-space cap.
                status = Path(f"/proc/{proc.pid}/status")
                try:status_text=status.read_text()
                except FileNotFoundError:status_text=''
                if status_text:
                    for line in status_text.splitlines():
                        if line.startswith("VmRSS:") and int(line.split()[1])*1024 > PROFILE['host_sampled_rss_bytes']:
                            raise RuntimeError("Host client exceeds sampled RSS limit")
                monitor.check(time.monotonic())
                time.sleep(.1)
            code=proc.returncode
            monitor.check(time.monotonic())
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
    finally:
        audit=cleanup(root,name,baseline,proc)
        if proc is not None:code=proc.poll()
        receipt=dict(stage=name,exit_code=code,error=error,jobs=audit['jobs'],
            child_pid=None if proc is None else proc.pid,child_reaped=proc is not None and code is not None,
            cleanup_errors=audit['cleanup_errors'],uncorrelated_new_jobs=audit['uncorrelated_new_jobs'],
            wall_seconds=time.monotonic()-started,maximum_journal_idle_seconds=monitor.max_journal_idle,
            maximum_monitor_poll_gap_seconds=monitor.max_poll_gap,monitor_target_poll_seconds=.1,
            hard_device_ACK_deadline_guaranteed=False)
        atomic_json(root/(name+'-audit.json'),receipt)
    if error or code!=0 or audit['cleanup_errors'] or len(audit['jobs'])!=1 or any(r['cancelled'] or r['phase']!='SUCCEEDED' for r in audit['jobs']):
        raise RuntimeError("Stage failed; original raw prefix and resource audit preserved: "+name)
    return receipt


def main():
    root=Path.cwd()
    def interrupted(signum,frame):raise RuntimeError("Supervisor interrupted: "+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (root.parent/'hardware.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            from source_gate import verify,PROFILE
            verify()
            if (root/'ACTIVE.json').exists():raise RuntimeError('Existing attempt; no implicit retry')
            active=query('jobs','--my-jobs')
            if any(j['status']['phase'] not in TERMINAL for j in active['items']):
                raise RuntimeError('Account has an active job; no overlap')
            receipt=stage(root,'run','run_hw.py',PROFILE['runtime_seconds'])
            verify()
            result=json.loads((root/'capture.json').read_bytes())
            lifecycle=json.loads((root/'runtime-lifecycle.json').read_bytes())
            if result['status']!='captured' or result['complete_epochs']!=[1,2,3] or not result['normal_stop'] or not result['held_observed'] or result['hold_release_rpc_seconds']>5 or not lifecycle['normal_stop'] or lifecycle['primary_error'] or lifecycle['stop_error']:
                raise RuntimeError('Complete capture/held RPC/normal-stop gate')
            atomic_json(root/'COMPLETE.json',dict(scope='Unreviewed finite136PE physical capture only',
                stages=[receipt],all_owned_jobs_released=True,mathematical_audit_passed=False,
                full_model=False,controller_acceptance=False))
            atomic_json(root/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
        except BaseException as exc:
            atomic_json(root/'FAILURE.json',dict(error=type(exc).__name__,message=str(exc)))
            raise


if __name__=='__main__':main()
