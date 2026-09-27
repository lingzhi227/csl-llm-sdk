"""Stage exact qualified autonomous mixed resident epoch sources and bounded fixture on hardware."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--simulation',required=True);p.add_argument('--reuse-from');a=p.parse_args()
if not re.fullmatch('[0-9]{3}',a.attempt):raise ValueError('attempt')
if not re.fullmatch('epoch-bank-sim-[0-9]{3}',a.simulation):raise ValueError('simulation')
if a.reuse_from is not None and not re.fullmatch('epoch-bank-hw-[0-9]{3}',a.reuse_from):raise ValueError('reuse attempt')
source=ROOT/'performance/probes/epoch_bank'
sim=ROOT/'performance/evidence'/a.simulation
receipt=json.loads((sim/'COMPLETE.json').read_text());assert receipt['result']['passed'] and receipt['result']['normal_stop'] and not receipt['physical']
sim_hash=json.loads((sim/'source-manifest.json').read_text())['files']['worker.csl']
files={n:(source/n).read_bytes() for n in ['worker.csl','controller.csl','run.py','prepare.py']}
files['layout.csl']=(sim/'source/layout.csl').read_bytes()
files['region.json']=(sim/'source/region.json').read_bytes()
assert hashlib.sha256(files['worker.csl']).hexdigest()==sim_hash
files['fp8_unpack.csl']=(ROOT/'performance/csl/fp8_unpack.csl').read_bytes()
files['fp8_dot.csl']=(ROOT/'performance/csl/fp8_dot.csl').read_bytes()
files['bf16_dot.csl']=(ROOT/'performance/csl/bf16_dot.csl').read_bytes()
files['REUSE.json']=(source/'REUSE.json').read_bytes()
for n in ['banks.json','bank-source.json']:files[n]=(sim/'source'/n).read_bytes()
frozen=json.loads((sim/'source-manifest.json').read_text())['files']
assert all(hashlib.sha256(v).hexdigest()==frozen[n] for n,v in files.items())
region=json.loads(files['region.json'])['plan'];w=region['mesh_width'];h=sum(region['blocks'])//w+1;pes=w*h
assert w==2 and h==4 and receipt['result']['application']==[w,h]
assert hashlib.sha256(files['layout.csl']).hexdigest()==json.loads((sim/'source-manifest.json').read_text())['files']['layout.csl']
assert hashlib.sha256(files['run.py']).hexdigest()==json.loads((sim/'source-manifest.json').read_text())['files']['run.py']
config=dict(application_pes=pes,application=[w,h],full_model=False,artifact_single_message_limit=8<<20,compiled_component_reuse=a.reuse_from is not None,
 compiler_timeout_seconds=600,run_timeout_seconds=300,simulation_protocol_source=a.simulation)
files['experiment.json']=(json.dumps(config,indent=2)+'\n').encode()
for n in ['backend.py','bounded_client.py','compile_hw.py','run_hw.py','supervise.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py']:
 files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['backend.py','supervise.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
 for n,v in files.items():
  item=tarfile.TarInfo(n);item.size=len(v);tar.addfile(item,io.BytesIO(v))
name='epoch-bank-hw-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
subprocess.run(['sh','/path/to/alcf-session.sh','host','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=30)
# Transfer the exact frozen small fixture directly between remotes, never onto Mac disk.
source_remote='/srv/model-storage/qwen38-singlewse/runs/'+a.simulation
session=['sh','/path/to/alcf-session.sh','host']
reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(source_remote)+' fixture.npz fixture.json'],stdout=subprocess.PIPE)
try:subprocess.run(session+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=30)
finally:
 reader.stdout.close()
 try:assert reader.wait(timeout=20)==0
 except subprocess.TimeoutExpired:reader.kill();reader.wait();raise
expected=json.loads((sim/'fixture.json').read_text())['fixture_sha256']
script='from pathlib import Path;import hashlib,json;p=Path('+repr(dest)+');m=json.loads((p/"fixture.json").read_text());assert hashlib.sha256((p/"fixture.npz").read_bytes()).hexdigest()==m["fixture_sha256"]=='+repr(expected)
subprocess.run(session+['python3 -c '+shlex.quote(script)],check=True,timeout=20)
if a.reuse_from:
 previous=ROOT/'performance/evidence'/a.reuse_from
 audit=json.loads((previous/'compile-audit.json').read_text());artifact=json.loads((previous/'artifact.json').read_text())
 assert audit['exit_code']==0 and audit['error'] is None and not audit['cleanup_errors'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in audit['jobs'])
 oldmanifest=json.loads((previous/'source-manifest.json').read_text())['files']
 csl_hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if n.endswith('.csl')}
 assert csl_hashes=={n:v for n,v in oldmanifest.items() if n.endswith('.csl')}
 sram=json.loads((previous/'sram.json').read_text());assert sram['passed'] and sram['application_pes']==pes
 reuse=dict(source_attempt=a.reuse_from,compiler_succeeded_and_released=True,application_pes=pes,csl_hashes=csl_hashes,artifact_sha256=artifact['sha256'],elf_hashes={Path(v['file']).name:v['sha256'] for v in sram['records']})
 script='source_name='+repr('/srv/qwen38-singlewse-hardware/'+a.reuse_from)+'\ndest_name='+repr(dest)+'\nreuse='+repr(reuse)+'\n'+'''from pathlib import Path
import json,hashlib,shutil
src=Path(source_name);dst=Path(dest_name);a=json.loads((src/'artifact.json').read_text());p=Path(a['artifact'])
assert p.stat().st_size<=8<<20 and hashlib.sha256(p.read_bytes()).hexdigest()==reuse['artifact_sha256']
shutil.copyfile(p,dst/p.name);a['artifact']=str(dst/p.name)
(dst/'out/bin').mkdir(parents=True)
for n,h in reuse['elf_hashes'].items():
 p=src/'out/bin'/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h;shutil.copyfile(p,dst/'out/bin'/n)
shutil.copyfile(src/'compile-audit.json',dst/'reused-compile-audit.json')
(dst/'artifact.json').write_text(json.dumps(a,indent=2)+'\\n');(dst/'reuse-artifact.json').write_text(json.dumps(reuse,indent=2)+'\\n')
'''
 subprocess.run(session+['python3 -c '+shlex.quote(script)],check=True,timeout=30)
out=ROOT/'evidence'/name;out.mkdir()
(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
(out/'staging.json').write_text(json.dumps(dict(remote=dest,simulation=a.simulation,source='performance/probes/epoch_bank',physical_dispatched=False),indent=2)+'\n')
print(json.dumps(dict(name=name,application_pes=pes,physical_dispatched=False)))
