"""Finite host-only metadata check, with no appliance or physical job creation."""
import fcntl,hashlib,json,os,resource,shutil,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILE=dict(seconds=300,address_space=4<<30,group_rss=1<<30,file_bytes=8<<20,run_bytes=1<<30,disk_reserve=32<<30,cpu_count=1)

def require(value,message):
    if not value:raise ValueError(message)

def verify():
    raw=(ROOT/'manifest.json').read_bytes();a=json.loads((ROOT/'audit-admission.json').read_bytes())
    require(a==dict(candidate=ROOT.name,manifest_sha256=hashlib.sha256(raw).hexdigest(),profile=PROFILE,audit_only=True,automatic_retry=False),'Exact numerical audit admission')
    for n,s in json.loads(raw)['files'].items():
        p=ROOT/n;require(p.is_file() and not p.is_symlink() and p.stat().st_size==s['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==s['sha256'],'Frozen checker identity')

def main():
    verify();require(not (ROOT/'audit-supervisor.json').exists(),'Already attempted')
    allowed=sorted(os.sched_getaffinity(0));cpu=allowed[0];started=time.monotonic();proc=None
    report=dict(status='running',profile=PROFILE,affinity=[cpu],peak_group_rss=0,native_task_samples=0)
    def save():
        temp=ROOT/'audit-supervisor.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(ROOT/'audit-supervisor.json')
    def child_limits():
        os.sched_setaffinity(0,{cpu});resource.setrlimit(resource.RLIMIT_AS,(PROFILE['address_space'],)*2)
        resource.setrlimit(resource.RLIMIT_FSIZE,(PROFILE['file_bytes'],)*2);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_CPU,(290,290))
    def stop_signal(signum,frame):raise RuntimeError('Numerical audit supervisor interrupted')
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    with (ROOT.parent/'hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save()
        try:
            require(shutil.disk_usage(ROOT).free>=PROFILE['disk_reserve'],'Host disk reserve')
            (ROOT/'audit-tmp').mkdir()
            env=dict(os.environ,TMPDIR=str(ROOT/'audit-tmp'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            command=[os.environ.get('CSL_AUDIT_PYTHON', 'python3'),'-B','audit_full.py']
            report['command']=command
            with (ROOT/'audit.log').open('x') as log:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_limits)
                report['owned_process_group']=proc.pid;save()
                while proc.poll() is None:
                    rows=subprocess.check_output(['ps','-e','-o','pid=,pgid=,rss='],text=True,timeout=5)
                    group=[list(map(int,line.split())) for line in rows.splitlines() if len(line.split())==3 and int(line.split()[1])==proc.pid]
                    rss=sum(row[2]*1024 for row in group);report['peak_group_rss']=max(report['peak_group_rss'],rss)
                    require(len(group)<=32 and rss<=PROFILE['group_rss'],'Owned group process/RSS bound')
                    for pid,_,_ in group:
                        try:
                            for task in (Path('/proc')/str(pid)/'task').iterdir():
                                require(os.sched_getaffinity(int(task.name))=={cpu},'Native CPU affinity');report['native_task_samples']+=1
                        except (FileNotFoundError,ProcessLookupError):pass
                    sizes=[p.stat().st_size for p in ROOT.rglob('*') if p.is_file()]
                    require(len(sizes)<=4096 and sum(sizes)<=PROFILE['run_bytes'],'Numerical audit directory bound')
                    require(shutil.disk_usage(ROOT).free>=PROFILE['disk_reserve'],'Host disk reserve')
                    require(time.monotonic()-started<PROFILE['seconds'],'Numerical audit deadline')
                    save();time.sleep(.5)
                require(proc.returncode==0 and json.loads((ROOT/'numerics.json').read_bytes())['status']=='passed','Full numerical audit failed')
            verify();report['status']='passed'
        except BaseException as exc:report.update(status='failed',message=str(exc),error=type(exc).__name__)
        finally:
            if proc is not None:
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    try:os.killpg(proc.pid,sig)
                    except ProcessLookupError:break
                    try:proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:pass
                rows=subprocess.check_output(['ps','-e','-o','pgid='],text=True,timeout=5)
                report['owned_group_quiescent']=str(proc.pid) not in rows.split()
                if not report['owned_group_quiescent']:report['status']='failed'
            report['seconds']=time.monotonic()-started;save()
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
