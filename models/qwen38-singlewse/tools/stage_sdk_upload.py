"""Qualify the full-artifact one-message profile with the installed SDK."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert re.fullmatch('[0-9]{3}',a.attempt)
name='sdk-upload-single-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
files={'bounded_client.py':(ROOT/'runtime/bounded_client.py').read_bytes(),'qualify.py':(ROOT/'tools/qualify_sdk_upload.py').read_bytes()}
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for n,v in files.items():
        entry=tarfile.TarInfo(n);entry.size=len(v);tar.addfile(entry,io.BytesIO(v))
session=['sh','/path/to/alcf-session.sh','host']
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=20)
script='''import os,resource,subprocess
from pathlib import Path
os.chdir('''+repr(dest)+''')
def caps():
 resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));resource.setrlimit(resource.RLIMIT_CPU,(60,60));resource.setrlimit(resource.RLIMIT_FSIZE,(256<<20,256<<20));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
with Path('qualification.log').open('x') as log:
 r=subprocess.run(['/opt/cerebras/venv/bin/python','qualify.py','--single-message'],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),preexec_fn=caps,timeout=90,stdout=log,stderr=subprocess.STDOUT)
print(Path('qualification.log').read_text());raise SystemExit(r.returncode)
'''
result=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=100)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'qualification.log').write_text(result.stdout+result.stderr)
print(result.stdout+result.stderr);assert result.returncode==0
(out/'COMPLETE.json').write_text(json.dumps(json.loads(result.stdout),indent=2)+'\n')
