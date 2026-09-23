"""One physical owner using the accepted job-correlation/release mechanism."""
from pathlib import Path
import fcntl,json,os,resource,shutil,signal,subprocess,sys,time
from runtime_gate import ROOT,verify
from runtime_capture import atomic_json
from runtime_monitor import Monitor
from hw00.watchdog import query,cleanup,TERMINAL

def group_members(pgid):
    p=subprocess.run(['ps','-eo','pid=,pgid='],capture_output=True,text=True,check=True,timeout=5)
    return [int(a) for line in p.stdout.splitlines() for a,b in [line.split()] if int(b)==pgid]

def terminate_owned(proc):
    if proc.poll() is None or group_members(proc.pid):
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    if group_members(proc.pid):
        os.killpg(proc.pid,signal.SIGKILL)
        if group_members(proc.pid):raise RuntimeError('Owned process group not released')

def main():
    # Unmodified query/cleanup helpers spawn csctl directly. Its Go runtime
    # needs the full virtual address limit, including every cleanup path.
    os.sched_setaffinity(0,{0});resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    def interrupted(signum,frame):raise RuntimeError('Physical guard interrupted '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (ROOT.parent/'hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);_,profiles,_=verify('runtime');profile=profiles['runtime']
        if (ROOT/'ACTIVE.json').exists():raise RuntimeError('Physical attempt already exists; never retry implicitly')
        if any(j['status']['phase'] not in TERMINAL for j in query('jobs','--my-jobs')['items']):raise RuntimeError('Active account job prevents overlap')
        baseline=query('jobs','--my-jobs','--all-states');atomic_json(ROOT/'run-jobs-before.json',baseline)
        started=time.monotonic();monitor=Monitor(ROOT,profile,started);proc=None;primary=None;peak=0;parent_peak=0;released=None
        def limits():
            os.sched_setaffinity(0,{0})
            for kind,value in [(resource.RLIMIT_AS,profile['address_space_bytes']),
                (resource.RLIMIT_CPU,profile['runtime_seconds']),(resource.RLIMIT_FSIZE,profile['max_journal_bytes']),
                (resource.RLIMIT_CORE,0)]:resource.setrlimit(kind,(value,value))
        try:
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HW00_JOB_CAPTURE=str(ROOT/'run.jobs'))
            with (ROOT/'run.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-B','runtime_entry.py'],cwd=ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
                atomic_json(ROOT/'ACTIVE.json',dict(phase='run',pid=proc.pid,pgid=proc.pid,guard_pid=os.getpid()))
                while proc.poll() is None:
                    monitor.check(time.monotonic())
                    parent_fields=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
                    parent_rss=int(parent_fields.get('VmRSS','0').split()[0])*1024;parent_peak=max(parent_peak,parent_rss)
                    if parent_rss>profile['guard_RSS_bytes'] or int(parent_fields.get('VmSwap','0').split()[0]):
                        raise RuntimeError('Physical guard RSS/swap')
                    if shutil.disk_usage(ROOT).free<profile['host_disk_reserve_bytes']:raise RuntimeError('Host disk reserve')
                    metadata_bytes=0
                    for p in ROOT.iterdir():
                        try:
                            if p.is_file():metadata_bytes+=p.stat().st_size
                        except FileNotFoundError:
                            # A progress/phase temporary may be atomically
                            # renamed after the directory entry was listed.
                            continue
                    # Raw files are charged before each write by the frozen
                    # Store and reported before the following SDK call. Avoid
                    # rescanning24000 immutable files at every0.1s sample.
                    if metadata_bytes+profile['max_raw_bytes']>profile['capture_directory_bytes']:
                        raise RuntimeError('Capture metadata plus reserved raw storage cap')
                    for name,cap in [('run.log',profile['host_log_bytes']),('run.jobs',65536),
                                     ('startup-stack.log',profile['startup_stack_bytes'])]:
                        path=ROOT/name
                        if path.exists() and path.stat().st_size>cap:raise RuntimeError('Bounded physical log '+name)
                    try:
                        raw=(Path('/proc')/str(proc.pid)/'status').read_text()
                        fields=dict(line.split(':',1) for line in raw.splitlines() if ':' in line)
                        rss=int(fields.get('VmRSS','0').split()[0])*1024;peak=max(peak,rss)
                        if rss>profile['RSS_bytes'] or int(fields.get('VmSwap','0').split()[0]):raise RuntimeError('Physical client RSS/swap')
                    except FileNotFoundError:pass
                    time.sleep(profile['parent_poll_seconds'])
                proc.wait()
                if proc.returncode!=0:raise RuntimeError('Physical client failed')
                monitor.check(time.monotonic())
        except BaseException as exc:primary=repr(exc)
        finally:
            released=cleanup(ROOT,'run',baseline,proc,terminate_fn=terminate_owned)
            atomic_json(ROOT/'run-audit.json',dict(profile=profile,child_pid=None if proc is None else proc.pid,
                exit_code=None if proc is None else proc.returncode,child_reaped=proc is None or proc.poll() is not None,
                remaining_group_members=[] if proc is None else group_members(proc.pid),
                wall_seconds=time.monotonic()-started,sampled_peak_RSS=peak,sampled_guard_peak_RSS=parent_peak,primary_error=primary,
                maximum_monitor_poll_gap_seconds=monitor.max_poll_gap,**released))
        if primary is not None or released['cleanup_errors'] or len(released['jobs'])!=1 or any(
            j['phase']!='SUCCEEDED' or not j['released'] or j['cancelled'] for j in released['jobs']):
            atomic_json(ROOT/'FAILURE.json',dict(primary_error=primary,cleanup= released));raise RuntimeError('Physical owner failed; original evidence preserved')
        verify('runtime');result=json.loads((ROOT/'capture.json').read_bytes());life=json.loads((ROOT/'runtime-lifecycle.json').read_bytes())
        if result['completed']!=[1,2,3] or not result['normal_stop'] or not life['normal_stop'] or life['primary_error'] or life['stop_error']:
            raise RuntimeError('Complete causal capture and normal stop required')
        atomic_json(ROOT/'COMPLETE.json',dict(scope='Complete original Layer3 raw capture, pending independent numerical audit',
            all_owned_jobs_released=True,mathematical_audit_passed=False,full_model=False,controller_acceptance=False))
        atomic_json(ROOT/'ACTIVE.json',dict(phase='complete',active_jobs=[]))

if __name__=='__main__':main()
