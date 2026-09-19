"""Release the one owned physical runtime after one finite diagnostic snapshot."""
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
            if result['status']!='captured' or not result['normal_stop'] or result['complete_epochs'] != []:
                raise RuntimeError('Finite observation and normal physical context exit required')
            if result['copies'] not in PROFILE['copies'] or result['copies']!=PROFILE['copies'][0] or result['launches']!=PROFILE['launches']:
                raise RuntimeError('Exact finite layer3 operation count')
            verify();atomic_json(ROOT/'COMPLETE.json',dict(scope='One-position original layer3 global plus 24 head observations and six finite FIFO prefixes; unconditional stop',stages=[receipt],
                    all_owned_jobs_released=True,mathematical_audit_passed=False,diagnostic_only=True,observation=result['observation'],complete_epochs=result['complete_epochs'],full_model=False))
            atomic_json(ROOT/'ACTIVE.json',dict(phase='released_observation_complete',active_jobs=[]))
        except BaseException as exc:
            atomic_json(ROOT/'FAILURE.json',dict(error=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':main()
