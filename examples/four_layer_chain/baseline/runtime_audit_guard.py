"""Bounded post-release CPU audit owner; never create or query an SDK runtime."""
from pathlib import Path
import fcntl,hashlib,json,os,resource,signal,subprocess,sys,time
from runtime_gate import ROOT,verify
from runtime_capture import atomic_json
from runtime_guard import terminate_owned,group_members

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def release_gate(profile):
    expected=dict(candidate=ROOT.name,runtime_manifest_sha256=digest(ROOT/'runtime-manifest.json'),
        run_audit_sha256=digest(ROOT/'run-audit.json'),dispatch_exit_sha256=digest(ROOT/'DISPATCH-EXIT.json'),
        capture_sha256=digest(ROOT/'capture.json'),profile=profile,original_runtime_session_reaped=True,
        controller_resource_release_confirmed=True,offline_only=True,SDK=False,automatic_retry=False)
    admission=json.loads((ROOT/'post-release-admission.json').read_bytes())
    if admission!=expected:raise ValueError('Exact controller post-release evidence and resource admission')
    dispatch=json.loads((ROOT/'DISPATCH-EXIT.json').read_bytes());run=json.loads((ROOT/'run-audit.json').read_bytes())
    complete=json.loads((ROOT/'COMPLETE.json').read_bytes())
    if not dispatch['guard_reaped'] or dispatch['guard_exit_code']!=0 or not complete['all_owned_jobs_released']:
        raise ValueError('Original runtime guard must be successfully reaped and released')
    if run['exit_code']!=0 or run['primary_error'] or run['cleanup_errors'] or run['uncorrelated_new_jobs'] or len(run['jobs'])!=1 or any(
        j['phase']!='SUCCEEDED' or not j['released'] or j['cancelled'] for j in run['jobs']):raise ValueError('Exact successful owned physical release')
    for pid in (dispatch['guard_pid'],dispatch['ssh_waiter_pid'],run['child_pid']):
        if (Path('/proc')/str(pid)).exists():raise ValueError('Original physical owner still exists')
    return expected

def main():
    _,profiles,_=verify('audit');profile=profiles['audit'];admission=release_gate(profile)
    os.sched_setaffinity(0,{0});resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30))
    def interrupted(signum,frame):raise RuntimeError('Offline guard interrupted '+str(signum))
    signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    def limits():
        os.sched_setaffinity(0,{0})
        for kind,value in [(resource.RLIMIT_AS,profile['address_space']),(resource.RLIMIT_CPU,profile['cpu_seconds']),
            (resource.RLIMIT_FSIZE,profile['file_bytes']),(resource.RLIMIT_CORE,0)]:resource.setrlimit(kind,(value,value))
    with (ROOT.parent/'hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (ROOT/'AUDIT-ATTEMPT.json').open('x') as f:json.dump(admission,f);f.flush();os.fsync(f.fileno())
        proc=None;error=None;started=time.monotonic();peak=0;parent_peak=0
        original_sizes={p.name:p.stat().st_size for p in ROOT.iterdir() if p.is_file()}
        try:
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
            with (ROOT/'audit.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-B','runtime_audit.py'],cwd=ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
                atomic_json(ROOT/'AUDIT-ACTIVE.json',dict(parent_pid=os.getpid(),child_pid=proc.pid))
                while proc.poll() is None:
                    if time.monotonic()-started>profile['seconds']:raise TimeoutError('Offline audit wall deadline')
                    if (ROOT/'audit.log').stat().st_size>profile['log_bytes']:raise ValueError('Offline log cap')
                    fields=dict(s.split(':',1) for s in Path('/proc/self/status').read_text().splitlines() if ':' in s)
                    parent_rss=int(fields.get('VmRSS','0').split()[0])*1024;parent_peak=max(parent_peak,parent_rss)
                    if parent_rss>profile['guard_RSS_bytes'] or int(fields.get('VmSwap','0').split()[0]):raise ValueError('Offline guard RSS/swap cap')
                    memory=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
                    if int(memory['MemAvailable'].split()[0])*1024<profile['host_available_memory_reserve_bytes']:raise ValueError('Offline host RAM reserve')
                    growth=0
                    for path in ROOT.iterdir():
                        try:
                            if path.is_file():growth+=max(0,path.stat().st_size-original_sizes.get(path.name,0))
                        except FileNotFoundError:continue
                    if growth>profile['directory_growth_bytes']:raise ValueError('Offline total output growth cap')
                    try:
                        raw=(Path('/proc')/str(proc.pid)/'status').read_text();fields=dict(s.split(':',1) for s in raw.splitlines() if ':' in s)
                        rss=int(fields.get('VmRSS','0').split()[0])*1024;peak=max(peak,rss)
                        if rss>profile['rss'] or int(fields.get('VmSwap','0').split()[0]):raise ValueError('Offline RSS/swap cap')
                    except FileNotFoundError:pass
                    time.sleep(.25)
                proc.wait()
                if proc.returncode!=0:raise RuntimeError('Offline audit child failed')
                if sum(max(0,p.stat().st_size-original_sizes.get(p.name,0)) for p in ROOT.iterdir() if p.is_file())>profile['directory_growth_bytes']:raise ValueError('Final offline output growth cap')
                verify('audit');release_gate(profile)
        except BaseException as exc:error=repr(exc)
        finally:
            cleanup_error=None
            try:
                if proc is not None:terminate_owned(proc)
            except BaseException as exc:cleanup_error=repr(exc)
            atomic_json(ROOT/'audit-supervisor.json',dict(error=error,cleanup_error=cleanup_error,
                exit_code=None if proc is None else proc.returncode,child_pid=None if proc is None else proc.pid,
                child_reaped=proc is None or proc.poll() is not None,remaining_group_members=[] if proc is None else group_members(proc.pid),
                wall_seconds=time.monotonic()-started,maximum_sampled_RSS=peak,maximum_sampled_guard_RSS=parent_peak,profile=profile,SDK=False))
        if error or cleanup_error:raise RuntimeError('Offline audit failed; original evidence preserved')

if __name__=='__main__':main()
