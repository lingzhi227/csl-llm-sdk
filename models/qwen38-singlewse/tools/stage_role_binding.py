"""No-job bounded ELF/host role binding audit on ALCF."""
from pathlib import Path
import argparse,io,json,hashlib,tarfile,subprocess,shlex,re
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--compiled',default='full-resident-hw-003');a=p.parse_args();assert re.fullmatch('[0-9]{3}',a.attempt)
assert re.fullmatch('full-resident-hw-[0-9]{3}',a.compiled)
name='resident-binding-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
files={n:(root/'runtime'/n).read_bytes() for n in ['role_binding.py','elf_inventory.py','resident_plan.py']}
files['source-manifest.json']=json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}).encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as t:
    for n,v in files.items():
        i=tarfile.TarInfo(n);i.size=len(v);t.addfile(i,io.BytesIO(v))
session=['sh','/path/to/alcf-session.sh','host']
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=20)
script='compiled='+repr('/srv/qwen38-singlewse-hardware/'+a.compiled)+'\n'+'''import os,resource,subprocess
from pathlib import Path
os.chdir('''+repr(dest)+''')
def caps():
 resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));resource.setrlimit(resource.RLIMIT_CPU,(120,120));resource.setrlimit(resource.RLIMIT_FSIZE,(32<<20,32<<20));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
with Path('run.log').open('x') as log:
 r=subprocess.run(['/opt/cerebras/venv/bin/python','role_binding.py','--compiled',compiled],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),preexec_fn=caps,timeout=180,stdout=log,stderr=subprocess.STDOUT)
print(Path('run.log').read_text());raise SystemExit(r.returncode)
'''
out=root/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
r=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,timeout=190)
(out/'run.log').write_text(r.stdout+r.stderr);print(r.stdout+r.stderr);assert r.returncode==0
(out/'COMPLETE.json').write_text(json.dumps(json.loads(r.stdout),indent=2)+'\n')
