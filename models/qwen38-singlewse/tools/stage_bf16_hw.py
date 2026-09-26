"""Freeze the exact140-row candidate and pipe its original fixture between remotes."""
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='resident-bf16-hw-001';dest='/srv/qwen38-singlewse-hardware/'+name
source='/srv/model-storage/qwen38-singlewse/runs/resident-bf16-sim-002'
frozen=json.loads((ROOT/'evidence/resident-bf16-sim-002/source-manifest.json').read_text())['files']
files={n:(ROOT/'experiments/resident_bf16'/n).read_bytes() for n in ['layout.csl','pe.csl','run.py','experiment.json']}
files['bf16_resident.csl']=(ROOT/'csl/bf16_resident.csl').read_bytes()
assert all(hashlib.sha256(v).hexdigest()==frozen[n] for n,v in files.items())
for n in ['backend.py','bounded_client.py','compile_hw.py','run_hw.py','supervise.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w') as archive:
    for n,v in files.items():
        item=tarfile.TarInfo(n);item.size=len(v);archive.addfile(item,io.BytesIO(v))
session=['sh','/path/to/alcf-session.sh','host']
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=20)
reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(source)+' fixture.npz fixture.json'],stdout=subprocess.PIPE)
try:subprocess.run(session+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=30)
finally:reader.stdout.close();assert reader.wait(timeout=20)==0
script='''import hashlib,json
from pathlib import Path
p=Path('''+repr(dest)+''')
m=json.loads((p/'fixture.json').read_text())
assert hashlib.sha256((p/'fixture.npz').read_bytes()).hexdigest()==m['fixture_sha256']
print(json.dumps(m))
'''
r=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=20)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'fixture.json').write_text(r.stdout)
print(json.dumps(dict(staged=True,physical_dispatched=False,remote=dest,fixture_sha256=json.loads(r.stdout)['fixture_sha256'])))
