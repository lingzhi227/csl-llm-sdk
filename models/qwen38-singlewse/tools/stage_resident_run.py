"""Freeze complete physical runtime sources and reuse the accepted full artifact."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--compiled',default='full-resident-hw-003');p.add_argument('--binding',default='resident-binding-002');a=p.parse_args();assert re.fullmatch('[0-9]{3}',a.attempt)
assert re.fullmatch('full-resident-hw-[0-9]{3}',a.compiled) and re.fullmatch('resident-binding-[0-9]{3}',a.binding)
name='resident-generation-hw-'+a.attempt;dest='/srv/qwen38-singlewse-hardware/'+name
compiled=ROOT/'evidence'/a.compiled;manifest=json.loads((compiled/'source-manifest.json').read_text())['files']
complete=json.loads((compiled/'COMPLETE.json').read_text());assert complete['all_owned_jobs_released']
assert all(j['phase']=='SUCCEEDED' and j['released'] for s in complete['stages'] for j in s['jobs'])
sram=json.loads((compiled/'sram.json').read_text());assert sram['passed'] and sram['application_pes']==870000 and sram['compressed_bytes']<=128<<20
binding=json.loads((ROOT/'evidence'/a.binding/'COMPLETE.json').read_text());assert binding['passed'] and binding['application_pes']==870000
if 'artifact_sha256' in binding:assert binding['artifact_sha256']==sram['artifact_sha256']
else:assert a.compiled=='full-resident-hw-003' and a.binding=='resident-binding-002'
transport=ROOT/'evidence/sdk-upload-single-002';assert json.loads((transport/'COMPLETE.json').read_text())['passed']
assert json.loads((transport/'source-manifest.json').read_text())['files']['bounded_client.py']==hashlib.sha256((ROOT/'runtime/bounded_client.py').read_bytes()).hexdigest()
parameters=json.loads((ROOT/'evidence/resident-parameter-audit-002/COMPLETE.json').read_text())
assert parameters['passed'] and parameters['small_original_tensors']==353
assert parameters['role_sha256']==binding['role_sha256'] and parameters['config_sha256']==binding['config_sha256']
for n,digest in parameters['source_manifest'].items():
    assert hashlib.sha256((ROOT/'runtime'/n).read_bytes()).hexdigest()==digest,n
files={n:(ROOT/'resident/generated'/n).read_bytes() for n in ['storage.csl','controller.csl','gdn.csl','attention.csl']}
files['layout.csl']=(ROOT/'resident/full-layout.csl').read_bytes()
for n in ['resident_matrix.csl','strip_coordinates-bf16-140.csl','fp8_codec.csl','fp8_activation.csl','qwen_math.csl','gdn_preprocess.csl','recurrent_value64.csl','attention_math.csl','kv_shard64.csl']:
    files[n]=(ROOT/'csl'/n).read_bytes()
for n in ['placement-resident-96-bf16-140.json','role-plan-96.json','device-matrices-bf16-140.json','tensors.json','hub.json','forward-program.json','rotary-frequencies.json']:
    files[n]=(ROOT/'configs'/n).read_bytes()
for n,v in files.items():assert hashlib.sha256(v).hexdigest()==manifest[n],n
for n in ['resident_run.py','resident_loader.py','resident_plan.py','weights.py','backend.py','bounded_client.py','job_capture.py','source_gate.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
files['supervise.py']=(ROOT/'runtime/supervise_resident.py').read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['full-acceptance-v1.json']=(ROOT/'configs/full-acceptance-v1.json').read_bytes()
files['full-acceptance-frozen.json']=(ROOT/'evidence/full-acceptance-frozen.json').read_bytes()
assert hashlib.sha256(files['full-acceptance-v1.json']).hexdigest()==json.loads(files['full-acceptance-frozen.json'])['sha256']
files['binding.json']=(ROOT/'evidence'/a.binding/'COMPLETE.json').read_bytes()
files['parameter-audit.json']=(ROOT/'evidence/resident-parameter-audit-002/COMPLETE.json').read_bytes()
files['sram.json']=(compiled/'sram.json').read_bytes()
reuse=dict(compiled_from='/srv/qwen38-singlewse-hardware/'+a.compiled,
    artifact_sha256=sram['artifact_sha256'],compiler_succeeded_and_released=True,
    compile_receipt_sha256=hashlib.sha256((compiled/'COMPLETE.json').read_bytes()).hexdigest(),
    csl_hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items() if n.endswith('.csl')},
    binding_audit=a.binding,upload_qualification='sdk-upload-single-002')
files['reuse-artifact.json']=(json.dumps(reuse,indent=2)+'\n').encode()
files['experiment.json']=(json.dumps(dict(application_pes=870000,application=[750,1160],full_resident=True,full_model=False,
    physical_candidate=True,host_as_gib=4,host_sampled_rss_mib=1024,runtime_server_memory_gib=4,runtime_server_millicpu=2000,
    artifact_single_message_limit_bytes=128<<20,deadline_seconds=9000))+'\n').encode()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for n,v in files.items():
        entry=tarfile.TarInfo(n);entry.size=len(v);tar.addfile(entry,io.BytesIO(v))
session=['sh','/path/to/alcf-session.sh','host']
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=40)
script='dest='+repr(dest)+'\n'+'''
import hashlib,json,os
from pathlib import Path
p=Path(dest);reuse=json.loads((p/'reuse-artifact.json').read_text());old=Path(reuse['compiled_from'])
assert hashlib.sha256((old/'COMPLETE.json').read_bytes()).hexdigest()==reuse['compile_receipt_sha256']
for name,digest in reuse['csl_hashes'].items():assert hashlib.sha256((old/name).read_bytes()).hexdigest()==digest
artifact=json.loads((old/'artifact.json').read_text());source=Path(artifact['artifact']);assert source.stat().st_size<=128<<20
h=hashlib.sha256()
with source.open('rb') as stream:
 for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
assert h.hexdigest()==artifact['sha256']==reuse['artifact_sha256']
target=p/source.name;os.link(source,target);artifact['artifact']=str(target)
(p/'artifact.json').write_text(json.dumps(artifact,indent=2)+chr(10))
print(json.dumps(dict(remote=dest,artifact_sha256=h.hexdigest(),artifact_bytes=source.stat().st_size,physical_dispatched=False)))
'''
r=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'staging.json').write_text(r.stdout)
print(r.stdout)
