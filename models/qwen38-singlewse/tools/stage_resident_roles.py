"""Freeze a new representative compile; no forward execution is submitted."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert re.fullmatch('[0-9]{3}',a.attempt)
name='resident-roles-hw-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
files={p.name:p.read_bytes() for p in (ROOT/'resident/generated').iterdir() if p.suffix in ['.csl','.json']}
for n in ['resident_matrix.csl','strip_coordinates-bf16-140.csl','fp8_codec.csl','fp8_activation.csl','qwen_math.csl','gdn_preprocess.csl','recurrent_value64.csl','attention_math.csl','kv_shard64.csl']:
    files[n]=(ROOT/'csl'/n).read_bytes()
for n in ['compile_hw.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py','artifact_stream.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
files['supervise.py']=(ROOT/'runtime/supervise_compile.py').read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['experiment.json']=(json.dumps(dict(application_pes=144,application=[12,12],resident_roles=True,full_model=False,compile_only=True))+'\n').encode()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w') as archive:
    for n,v in files.items():
        entry=tarfile.TarInfo(n);entry.size=len(v);archive.addfile(entry,io.BytesIO(v))
subprocess.run(['sh','/path/to/alcf-session.sh','host','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=30)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
print(json.dumps(dict(name=name,remote=dest,compile_only=True,dispatched=False)))
