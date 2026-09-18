"""Release the one owned physical runtime before any expensive numerical audit."""
import fcntl,json,signal
from source_gate import ROOT,PROFILE,verify
from hw00.watchdog import query,stage,TERMINAL
from hw00.store import atomic_json

def main():
    def interrupted(signum,frame):raise RuntimeError('Runtime supervisor interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (ROOT.parent/'hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        if (ROOT/'ACTIVE.json').exists():raise RuntimeError('Physical candidate already attempted')
        if any(j['status']['phase'] not in TERMINAL for j in query('jobs','--my-jobs')['items']):raise RuntimeError('Account already has active job')
        try:
            receipt=stage(ROOT,'run','run_hw.py',PROFILE['runtime_seconds'])
            result=json.loads((ROOT/'capture.json').read_bytes())
            if result['status']!='captured' or not result['normal_stop'] or result['complete_epochs']!=[1,2,3,4]:
                raise RuntimeError('Four epochs and normal physical context exit required')
            verify();atomic_json(ROOT/'COMPLETE.json',dict(scope='Full original layer3 physical capture only',stages=[receipt],
                    all_owned_jobs_released=True,mathematical_audit_passed=False,full_model=False))
            atomic_json(ROOT/'ACTIVE.json',dict(phase='released_capture_complete_pending_numerics',active_jobs=[]))
        except BaseException as exc:
            atomic_json(ROOT/'FAILURE.json',dict(error=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':main()
