"""Fresh bounded recurrent/transport experiment; never modifies existing runs."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SESSION=['sh','/path/to/alcf-session.sh','host']
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--experiment',choices=['recurrent_head','resident_bus','qwen_math','gdn_head','bank_chain'],default='recurrent_head');a=p.parse_args()
assert re.fullmatch(r'[0-9]{3}',a.attempt)
name=a.experiment.replace('_','-')+'-hw-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
files={n:(ROOT/'experiments'/a.experiment/n).read_bytes() for n in ['layout.csl','pe.csl','prepare.py','run.py','experiment.json']}
libraries={'recurrent_head':['recurrent_value64.csl'],
    'resident_bus':['fp8_matrix_block.csl','fp8_codec.csl','fp8_activation.csl'],
    'qwen_math':['qwen_math.csl'],
    'gdn_head':['qwen_math.csl','gdn_preprocess.csl','recurrent_value64.csl'],
    'bank_chain':['qwen_math.csl','fp8_matrix_block.csl','fp8_codec.csl','fp8_activation.csl']}[a.experiment]
for n in libraries:files[n]=(ROOT/'csl'/n).read_bytes()
for n in ['backend.py','bounded_client.py','compile_hw.py','run_hw.py','supervise.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as archive:
    for n,v in files.items():
        info=tarfile.TarInfo(n);info.size=len(v);archive.addfile(info,io.BytesIO(v))
subprocess.run(SESSION+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=30)
script='dest='+repr(dest)+'\n'+'''
import os,resource,subprocess,json
from pathlib import Path
os.chdir(dest)
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
 resource.setrlimit(resource.RLIMIT_CPU,(60,60))
 resource.setrlimit(resource.RLIMIT_FSIZE,(64<<20,64<<20))
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with open('prepare.log','x') as log:
 r=subprocess.run(['/opt/cerebras/venv/bin/python','prepare.py'],env=env,preexec_fn=limits,stdout=log,stderr=subprocess.STDOUT,timeout=90)
assert r.returncode==0, r.returncode
print(json.dumps(dict(remote=dest,fixture=json.loads(Path('fixture.json').read_text()),physical_dispatched=False)))
'''
r=subprocess.run(SESSION+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=100)
out=ROOT/'evidence'/name;out.mkdir()
(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'staging.json').write_text(r.stdout)
print(r.stdout)
