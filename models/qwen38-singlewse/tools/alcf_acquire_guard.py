"""Bound task-owned checkpoint acquisition independently of hardware jobs."""
import json,os,resource,signal,subprocess,sys,time
from pathlib import Path
from source_gate import verify

def limits():
    resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8<<30,8<<30))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    resource.setrlimit(resource.RLIMIT_CPU,(5400,5400))
    os.nice(10)

def main():
    verify();started=time.monotonic();child=None;error=None;code=None
    try:
        with Path('download.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u','acquire.py','--hub','hub.json',
                '--root','/srv/qwen38-singlewse-hardware/model','--storage-tier','alcf','--max-mib-per-second','16'],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,preexec_fn=limits)
            Path('ACTIVE.json').write_text(json.dumps(dict(pid=child.pid,wall_limit_seconds=5400,as_limit_mib=512,max_file_gib=8,bandwidth_limit_mib_s=16,physical_job=False))+'\n')
            code=child.wait(timeout=5400)
            if code:raise RuntimeError('Download worker exited '+str(code))
    except BaseException as exc:error=repr(exc)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        result=dict(passed=error is None,exit_code=code,error=error,seconds=time.monotonic()-started,physical_job=False,
                    child_exited=child is None or child.poll() is not None)
        Path('receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        Path('ACTIVE.json').write_text(json.dumps(dict(phase='complete' if error is None else 'failed',active_processes=[]))+'\n')
    if error:raise RuntimeError(error)

if __name__=='__main__':main()
