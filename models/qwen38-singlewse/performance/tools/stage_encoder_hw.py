"""Stage direct encoder/full dynamic group physical qualification after bounded simulation."""
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
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--simulation',required=True);a=p.parse_args()
if not re.fullmatch('[0-9]{3}',a.attempt):raise ValueError('attempt')
if not re.fullmatch('fp8-encoder-sim-[0-9]{3}',a.simulation):raise ValueError('simulation')
source=ROOT/'performance/probes/fp8_encoder'
sim=ROOT/'performance/evidence'/a.simulation
receipt=json.loads((sim/'COMPLETE.json').read_text());assert receipt['result']['passed'] and receipt['result']['normal_stop'] and not receipt['physical']
sim_hash=json.loads((sim/'source-manifest.json').read_text())['files']['pe.csl']
files={n:(source/n).read_bytes() for n in ['layout.csl','pe.csl','run.py','prepare.py']}
assert hashlib.sha256(files['pe.csl']).hexdigest()==sim_hash
for n in ['fp8_encode.csl','fp8_quantize.csl']:files[n]=(ROOT/'performance/csl'/n).read_bytes()
for n in ['fp8_codec.csl','fp8_activation.csl']:files[n]=(ROOT/'csl'/n).read_bytes()
files['encoder_oracle.py']=(ROOT/'performance/tools/prove_fp8_encoder.py').read_bytes()
frozen=json.loads((sim/'source-manifest.json').read_text())['files']
assert all(hashlib.sha256(v).hexdigest()==frozen[n] for n,v in files.items())
pes=2
assert receipt['result']['simulator_subset'] and receipt['result']['fixture_direct_words']==53725
assert hashlib.sha256(files['layout.csl']).hexdigest()==json.loads((sim/'source-manifest.json').read_text())['files']['layout.csl']
assert hashlib.sha256(files['run.py']).hexdigest()==json.loads((sim/'source-manifest.json').read_text())['files']['run.py']
config=dict(application_pes=pes,application=[2,1],full_model=False,
 compiler_timeout_seconds=600,run_timeout_seconds=300,simulation_protocol_source=a.simulation)
files['experiment.json']=(json.dumps(config,indent=2)+'\n').encode()
for n in ['backend.py','bounded_client.py','compile_hw.py','run_hw.py','supervise.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py']:
 files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
 for n,v in files.items():
  item=tarfile.TarInfo(n);item.size=len(v);tar.addfile(item,io.BytesIO(v))
name='fp8-encoder-hw-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
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
out=ROOT/'evidence'/name;out.mkdir()
(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
(out/'staging.json').write_text(json.dumps(dict(remote=dest,simulation=a.simulation,source='performance/probes/fp8_encoder',physical_dispatched=False),indent=2)+'\n')
print(json.dumps(dict(name=name,application_pes=pes,physical_dispatched=False)))
