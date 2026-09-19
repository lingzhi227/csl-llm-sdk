"""One four-PE fixture compile, reusing the accepted owner-correlated job cleanup."""
import fcntl,json,signal
from source_gate import ROOT,verify,PROFILE
from hw00.watchdog import query,stage,TERMINAL
from hw00.store import atomic_json

def main():
    def interrupted(signum,frame):raise RuntimeError('Compile supervisor interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (ROOT.parent/'hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        if (ROOT/'ACTIVE.json').exists():raise RuntimeError('Compile candidate already attempted')
        active=query('jobs','--my-jobs')
        if any(j['status']['phase'] not in TERMINAL for j in active['items']):raise RuntimeError('Account has active jobs')
        try:
            receipt=stage(ROOT,'compile','compile_hw.py',PROFILE['compile_seconds'])
            verify()
            if not json.loads((ROOT/'sram.json').read_bytes())['passed'] or not json.loads((ROOT/'placement.json').read_bytes())['passed']:
                raise RuntimeError('Physical compile gates failed')
            atomic_json(ROOT/'COMPLETE.json',dict(scope='Four-PE original-kernel archive/alias compile only; physical runtime requires separate admission',stages=[receipt],all_owned_jobs_released=True,runtime_created=False,full_model=False))
            atomic_json(ROOT/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
        except BaseException as exc:
            atomic_json(ROOT/'FAILURE.json',dict(error=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':main()
