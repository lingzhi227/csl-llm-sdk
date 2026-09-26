"""Reuse unchanged qualified CSL/artifact; retry only bounded SDK host transfers."""
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST='/srv/qwen38-singlewse-hardware/fp8-matrix-hw-003'
SESSION=['sh','/path/to/alcf-session.sh','host']
reuse=json.loads((ROOT/'evidence/fp8-matrix-hw-002/reuse-artifact.json').read_text())
files={n:(ROOT/'experiments/fp8_matrix'/n).read_bytes() for n in ['layout.csl','pe.csl','run.py','prepare.py','experiment.json']}
for n in ['fp8_matrix_block.csl','fp8_codec.csl','fp8_activation.csl']:files[n]=(ROOT/'csl'/n).read_bytes()
assert {n:hashlib.sha256(v).hexdigest() for n,v in files.items() if n.endswith('.csl')}==reuse['csl_hashes']
for n in ['backend.py','bounded_client.py','compile_hw.py','run_hw.py','supervise.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['reuse-artifact.json']=(json.dumps(reuse,indent=2)+'\n').encode()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as t:
    for n,v in files.items():
        i=tarfile.TarInfo(n);i.size=len(v);t.addfile(i,io.BytesIO(v))
subprocess.run(SESSION+['mkdir '+shlex.quote(DEST)+' && tar -xf - -C '+shlex.quote(DEST)],input=buf.getvalue(),check=True,timeout=30)
script='dest='+repr(DEST)+'\n'+'''
from pathlib import Path
import hashlib,json,os
p=Path(dest);r=json.loads((p/'reuse-artifact.json').read_text());old=Path(r['compiled_from'])
assert {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in old.glob('*.csl')}==r['csl_hashes']
artifact=json.loads((old/'artifact.json').read_text());source=Path(artifact['artifact'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==r['artifact_sha256']
target=p/source.name;os.link(source,target);artifact['artifact']=str(target)
(p/'artifact.json').write_text(json.dumps(artifact,indent=2)+chr(10))
(p/'out/bin').mkdir(parents=True)
for elf in (old/'out/bin').glob('*.elf'):os.link(elf,p/'out/bin'/elf.name)
for name in ['fixture.npz','fixture.json','oracle-qualification.json']:
    os.link(old/name,p/name)
assert hashlib.sha256((p/'fixture.npz').read_bytes()).hexdigest()==json.loads((p/'fixture.json').read_text())['fixture_sha256']
print(json.dumps(dict(remote=str(p),changed='Host transfers bounded below16MiB; stage logging and faulthandler',csl_unchanged=True,criteria_unchanged=True,physical_dispatched=False)))
'''
r=subprocess.run(SESSION+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
out=ROOT/'evidence/fp8-matrix-hw-003';out.mkdir()
(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'staging.json').write_text(r.stdout)
print(r.stdout)
