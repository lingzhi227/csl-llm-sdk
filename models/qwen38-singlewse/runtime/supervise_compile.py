"""Bounded compile-only role admission under the shared hardware lock."""
import fcntl,json,signal
from pathlib import Path
from runtime.lifecycle import stage,query,TERMINAL
from runtime.store import atomic_json
from source_gate import verify

def interrupted(number,frame):raise RuntimeError('Supervisor signal '+str(number))

def main():
    root=Path.cwd();signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    with Path('/srv/cerebras-hardware/hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        assert not (root/'ACTIVE.json').exists()
        active=query('jobs','--my-jobs')
        assert all(j['status']['phase'] in TERMINAL for j in active['items']),'Another account job is active'
        try:
            config=json.loads((root/'experiment.json').read_text());full=config.get('full_resident',False)
            assert not full or (config['application_pes']==870000 and config['application']==[750,1160])
            receipt=stage(root,'compile','compile_resident.py' if full else 'compile_hw.py',3600 if full else 600)
            verify();atomic_json(root/'COMPLETE.json',dict(scope='Complete resident layout compile only' if full else 'Representative complete-role SRAM compile only',full_model=False,physical_execution=False,stages=[receipt],all_owned_jobs_released=True))
            atomic_json(root/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
        except BaseException as error:
            atomic_json(root/'FAILURE.json',dict(error=type(error).__name__,message=str(error)));raise

if __name__=='__main__':main()
