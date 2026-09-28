"""Dispatch a single bounded metadata query for the retained mixer001 artifact."""
import ast, hashlib, io, json, shlex, subprocess, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
name='mixer-artifact-probe-001';out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen probe already exists')
files={}
for n in ['probe_mixer_artifact.py','progress_lifecycle.py','backend.py','progress.py']:
    files[n]=(ROOT/'performance/runtime'/n).read_bytes()
for n in ['job_capture.py','source_gate.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['supervise.py']=b'''import fcntl,json,signal
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
'''
files['stager.py']=Path(__file__).read_bytes()
for n,b in files.items():ast.parse(b)
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
payload=io.BytesIO()
with tarfile.open(fileobj=payload,mode='w') as tar:
    for n,b in files.items():
        info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
remote='/srv/qwen38-singlewse-hardware/'+name
connection=['sh','/path/to/alcf-session.sh','host']
subprocess.run(connection+['mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
out.mkdir()
for n,b in files.items():
    p=out/('source-manifest.json' if n=='source-manifest.json' else 'source/'+n);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
script='root_name='+repr(remote)+'\n'+'''import json,os,subprocess
from pathlib import Path
root=Path(root_name);os.chdir(root)
with (root/'supervisor.log').open('x') as log:
 p=subprocess.Popen(['/opt/cerebras/venv/bin/python','-u','supervise.py'],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(pid=p.pid,root=str(root),metadata_only=True)
(root/'SUPERVISOR.json').write_text(json.dumps(receipt)+'\\n');print(json.dumps(receipt))
'''
result=subprocess.run(connection+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
(out/'dispatch.json').write_text(result.stdout);print(result.stdout.strip())
