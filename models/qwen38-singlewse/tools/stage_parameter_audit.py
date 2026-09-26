"""Fresh bounded audit of resident parameter reads; no physical job."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--resident',required=True);a=p.parse_args()
assert re.fullmatch('[0-9]{3}',a.attempt) and re.fullmatch('resident-generation-hw-[0-9]{3}',a.resident)
name='resident-parameter-audit-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
resident='/srv/qwen38-singlewse-hardware/'+a.resident
files={'qualify.py':(ROOT/'tools/qualify_resident_parameters.py').read_bytes()}
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for n,v in files.items():
        e=tarfile.TarInfo(n);e.size=len(v);tar.addfile(e,io.BytesIO(v))
session=['sh','/path/to/alcf-session.sh','host']
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=20)
script='dest='+repr(dest)+'\nresident='+repr(resident)+'\n'+'''
import os,resource,subprocess
from pathlib import Path
os.chdir(dest);assert Path('/srv/qwen38-singlewse-hardware/model/COMPLETE.json').exists()
def caps():
 resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));resource.setrlimit(resource.RLIMIT_CPU,(300,300));resource.setrlimit(resource.RLIMIT_FSIZE,(32<<20,32<<20));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
with Path('run.log').open('x') as log:
 r=subprocess.run(['/opt/cerebras/venv/bin/python','qualify.py','--resident',resident],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONPATH=resident),preexec_fn=caps,timeout=300,stdout=log,stderr=subprocess.STDOUT)
print(Path('run.log').read_text());raise SystemExit(r.returncode)
'''
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
r=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=310)
(out/'run.log').write_text(r.stdout+r.stderr);print(r.stdout+r.stderr);assert r.returncode==0
(out/'COMPLETE.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
