"""Stage simulator-qualified route transitions; no allocation until dispatch."""
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
if not re.fullmatch('[0-9]{3}',a.attempt) or not re.fullmatch('route-epoch-sim-[0-9]{3}',a.simulation):raise ValueError('Attempt identity')
sim=ROOT/'performance/evidence'/a.simulation
receipt=json.loads((sim/'COMPLETE.json').read_text())
assert receipt['result']['passed'] and receipt['result']['normal_stop'] and not receipt['physical']
assert receipt['result']['application']==[4,5] and receipt['result']['all_bank_sentinels_retained']
assert json.loads((sim/'workstation-release.json').read_text())['workstation_released']
frozen=json.loads((sim/'source-manifest.json').read_text())['files'];files={}
for n,h in frozen.items():
 if n.endswith('.csl') or n in ['run.py','region.json']:
  b=(sim/'source'/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h;files[n]=b
files['experiment.json']=(json.dumps(dict(application_pes=20,application=[4,5],full_model=False,artifact_single_message_limit=8<<20,
 compiler_timeout_seconds=600,run_timeout_seconds=300,simulation_protocol_source=a.simulation,
 scope='Synthetic257-word dependent packet feedback through20PE alternating route/color/queue epochs, with35256B sentinel storage perPE; no neural arithmetic or full model.'),indent=2)+'\n').encode()
for n in ['bounded_client.py','run_hw.py','job_capture.py','source_gate.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['backend.py','supervise.py','check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
for n in ['check_sram.py','placement.py']:assert hashlib.sha256(files[n]).hexdigest()==frozen[n]
files['compile_hw.py']=(ROOT/'performance/runtime/compile_component.py').read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
name='route-epoch-hw-'+a.attempt;destination='/srv/qwen38-singlewse-hardware/'+name
out=ROOT/'evidence'/name
if out.exists():raise ValueError('Frozen hardware attempt exists')
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tar:
 for n,b in files.items():
  item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
subprocess.run(['sh','/path/to/alcf-session.sh','host','mkdir '+shlex.quote(destination)+' && tar -xf - -C '+shlex.quote(destination)],input=buffer.getvalue(),check=True,timeout=30)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'staging.json').write_text(json.dumps(dict(remote=destination,simulation=a.simulation,source='performance/probes/route_epoch',physical_dispatched=False),indent=2)+'\n')
print(json.dumps(dict(name=name,application_pes=20,physical_dispatched=False)))
