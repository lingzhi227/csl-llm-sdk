"""Stage a bounded independent norm diagnostic, without reserving hardware."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert re.fullmatch('[0-9]{3}',a.attempt)
name='norm-accuracy-hw-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
files={f.name:f.read_bytes() for f in (ROOT/'experiments/norm_accuracy').iterdir() if f.is_file()}
files['tensors.json']=(ROOT/'configs/tensors.json').read_bytes()
for n in ['backend.py','bounded_client.py','compile_hw.py','run_hw.py','supervise.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py']:
 files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
 for n,v in files.items():
  item=tarfile.TarInfo(n);item.size=len(v);tar.addfile(item,io.BytesIO(v))
session=['sh','/path/to/alcf-session.sh','host']
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=30)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
script='dest='+repr(dest)+'\n'+'''
import os,resource,subprocess,json
from pathlib import Path
os.chdir(dest)
def caps():
 resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
 resource.setrlimit(resource.RLIMIT_FSIZE,(256<<20,256<<20));resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.nice(10)
with Path('prepare.log').open('x') as log:
 r=subprocess.run(['/opt/cerebras/venv/bin/python','prepare.py'],env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),preexec_fn=caps,timeout=150,stdout=log,stderr=subprocess.STDOUT)
assert r.returncode==0,Path('prepare.log').read_text()[-3000:]
print(json.dumps(dict(remote=dest,fixture=json.loads(Path('fixture.json').read_text()),physical_dispatched=False)))
'''
r=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=160)
(out/'staging.json').write_text(r.stdout);print(r.stdout)
