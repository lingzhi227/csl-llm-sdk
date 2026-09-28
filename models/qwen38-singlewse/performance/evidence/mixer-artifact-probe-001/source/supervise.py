import fcntl,json,signal
from pathlib import Path
from progress_lifecycle import stage,query,TERMINAL
from runtime.store import atomic_json
from source_gate import verify
def interrupted(number,frame):raise RuntimeError('Supervisor signal '+str(number))
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
root=Path.cwd()
with Path('/srv/cerebras-hardware/hardware.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 assert not (root/'ACTIVE.json').exists()
 assert all(j['status']['phase'] in TERMINAL for j in query('jobs','--my-jobs')['items'])
 try:
  receipt=stage(root,'compile','probe_mixer_artifact.py',180)
  verify();atomic_json(root/'COMPLETE.json',dict(metadata_only=True,stages=[receipt],all_owned_jobs_released=True))
 except BaseException as error:
  atomic_json(root/'FAILURE.json',dict(error=type(error).__name__,message=str(error)));raise
