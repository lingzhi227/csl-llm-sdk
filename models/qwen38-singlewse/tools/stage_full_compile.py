"""Freeze complete layout compile inputs; respects the shared admission gate."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--role-compile',default='resident-roles-hw-005');a=p.parse_args();assert re.fullmatch('[0-9]{3}',a.attempt)
assert re.fullmatch('resident-roles-hw-[0-9]{3}',a.role_compile)
name='full-resident-hw-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
role_gate=json.loads((ROOT/'evidence'/a.role_compile/'sram.json').read_text());assert role_gate['passed'] and role_gate['application_pes']==144
role_sources=json.loads((ROOT/'evidence'/a.role_compile/'source-manifest.json').read_text())['files']
transport=ROOT/'evidence/compiler-transport-qualify-003'
assert json.loads((transport/'COMPLETE.json').read_text())['passed']
assert json.loads((transport/'source-manifest.json').read_text())['files']['bounded_compiler.py']==hashlib.sha256((ROOT/'runtime/bounded_compiler.py').read_bytes()).hexdigest()
files={n:(ROOT/'resident/generated'/n).read_bytes() for n in ['storage.csl','controller.csl','gdn.csl','attention.csl']}
files['layout.csl']=(ROOT/'resident/full-layout.csl').read_bytes()
for n in ['resident_matrix.csl','strip_coordinates-bf16-140.csl','fp8_codec.csl','fp8_activation.csl','qwen_math.csl','gdn_preprocess.csl','recurrent_value64.csl','attention_math.csl','kv_shard64.csl']:
    files[n]=(ROOT/'csl'/n).read_bytes()
for n,v in files.items():
    if n.endswith('.csl') and n!='layout.csl':assert hashlib.sha256(v).hexdigest()==role_sources[n],n
for n in ['compile_resident.py','job_capture.py','source_gate.py','elf_inventory.py','artifact_stream.py','resident_plan.py','bounded_compiler.py','bounded_client.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
files['supervise.py']=(ROOT/'runtime/supervise_compile.py').read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
for n in ['placement-resident-96-bf16-140.json','role-plan-96.json','device-matrices-bf16-140.json','tensors.json','hub.json','forward-program.json','rotary-frequencies.json']:
    files[n]=(ROOT/'configs'/n).read_bytes()
files['full-layout-source.json']=(ROOT/'evidence/full-layout-source.json').read_bytes()
assert hashlib.sha256(files['layout.csl']).hexdigest()==json.loads(files['full-layout-source.json'])['source_sha256']
files['experiment.json']=(json.dumps(dict(application_pes=870000,application=[750,1160],full_resident=True,full_model=False,compile_only=True,
    compiler_server_memory_gib=16,compiler_server_millicpu=4000,host_as_gib=4,host_sampled_rss_mib=1024,compile_deadline_seconds=3600))+'\n').encode()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w') as archive:
    for n,v in files.items():
        entry=tarfile.TarInfo(n);entry.size=len(v);archive.addfile(entry,io.BytesIO(v))
subprocess.run(['sh','/path/to/alcf-session.sh','host','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=40)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
print(json.dumps(dict(name=name,remote=dest,compile_only=True,dispatched=False,source_bytes=len(stream.getvalue()))))
