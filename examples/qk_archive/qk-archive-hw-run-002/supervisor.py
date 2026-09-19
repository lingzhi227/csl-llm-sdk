"""One four-PE physical fixture runtime, reusing the accepted owner-correlated job cleanup."""
import fcntl,json,signal
from source_gate import ROOT,verify,PROFILE
from hw00.watchdog import query,stage,TERMINAL
from hw00.store import atomic_json

def main():
    def interrupted(signum,frame):raise RuntimeError('Physical fixture supervisor interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    with (ROOT.parent/'hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        if (ROOT/'ACTIVE.json').exists():raise RuntimeError('Physical fixture candidate already attempted')
        active=query('jobs','--my-jobs')
        if any(j['status']['phase'] not in TERMINAL for j in active['items']):raise RuntimeError('Account has active jobs')
        try:
            receipt=stage(ROOT,'run','run_hw.py',PROFILE['runtime_seconds'])
            verify()
            result=json.loads((ROOT/'result.json').read_bytes())
            if not (result['status']=='passed' and result['normal_stop'] and result['copies']==13 and result['launches']==6 and result['host_slot_bytes']==32768 and result['journal_events']==43 and len(result['epochs'])==2 and all(e['status']=='passed' for e in result['epochs']) and result['negative']['status']=='passed' and result['negative']['actual_rejected_calls']==61 and result['negative']['actual_positive_calls']==172):
                raise RuntimeError('Complete physical two-reset archive/alias contract')
            atomic_json(ROOT/'COMPLETE.json',dict(scope='Four-PE synthetic original-kernel alias/archive physical qualification',stages=[receipt],all_owned_jobs_released=True,simulator=False,full_model=False))
            atomic_json(ROOT/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
        except BaseException as exc:
            atomic_json(ROOT/'FAILURE.json',dict(error=type(exc).__name__,message=str(exc)));raise

if __name__=='__main__':main()
