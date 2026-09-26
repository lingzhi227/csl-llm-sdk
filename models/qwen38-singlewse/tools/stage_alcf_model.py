"""Stage bounded official checkpoint acquisition; no hardware allocation."""
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];dest='/srv/qwen38-singlewse-hardware/acquisition-001'
files={n:(ROOT/path).read_bytes() for n,path in {'acquire.py':'tools/acquire.py','guard.py':'tools/alcf_acquire_guard.py','hub.json':'configs/hub.json','source_gate.py':'runtime/source_gate.py'}.items()}
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w') as archive:
    for n,v in files.items():
        entry=tarfile.TarInfo(n);entry.size=len(v);archive.addfile(entry,io.BytesIO(v))
session=['sh','/path/to/alcf-session.sh','host']
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=30)
script='''import json,os,subprocess
from pathlib import Path
root=Path('''+repr(dest)+''');os.chdir(root)
from source_gate import verify
verify()
with Path('supervisor.log').open('x') as log:
 p=subprocess.Popen(['python3','-u','guard.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,root=str(root),physical_job=False)
Path('SUPERVISOR.json').write_text(json.dumps(r)+'\\n');print(json.dumps(r))
'''
r=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=20)
out=ROOT/'evidence/alcf-acquisition-001';out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(r.stdout)
print(r.stdout)
