"""Freeze the complete original MLP physical qualification without dispatch.

Source is generated from the resident-stage lowering. Large original bank and
oracle payloads travel directly between the two remote stores, never Mac disk.
"""
import argparse,hashlib,io,json,re,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'));sys.path.insert(0,str(Path(__file__).parent))
from stage_layer_mlp_compile import build


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--reuse-download');parser.add_argument('--reuse-admitted');parser.add_argument('--reuse-payload');parser.add_argument('--shared-inputs',action='store_true');parser.add_argument('--norm-bridge',action='store_true');parser.add_argument('--reference',default='layer-mlp-reference-001');args=parser.parse_args()
    if args.reuse_payload and (args.reuse_download or not re.fullmatch('layer-mlp-hw-[0-9]{3}',args.reuse_payload)):
        raise ValueError('Payload-only reuse requires one valid attempt and a fresh compiler')
    if args.reuse_admitted and (not args.reuse_download or not re.fullmatch('layer-mlp-hw-[0-9]{3}',args.reuse_admitted)):
        raise ValueError('Admitted image reuse requires original compiler service provenance')
    if not re.fullmatch('[0-9]{3}',args.attempt):raise ValueError('Attempt')
    name='layer-mlp-hw-'+args.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    if not re.fullmatch('layer-mlp-reference-[0-9]{3}',args.reference):raise ValueError('Reference identity')
    reference=ROOT/'performance/evidence'/args.reference
    receipt=json.loads((reference/'COMPLETE.json').read_text());meta=json.loads((reference/'fixture.json').read_text())
    if not receipt['passed'] or not json.loads((reference/'workstation-release.json').read_text())['workstation_released']:
        raise ValueError('Completed and released independent reference required')
    expected=receipt['fixture_sha256']
    if hashlib.sha256((reference/'fixture.json').read_bytes()).hexdigest()!=expected:raise ValueError('Fixture identity')
    if bool(meta.get('norm_bridge'))!=args.norm_bridge:raise ValueError('Reference graph mismatch')
    files,width,height,profiles=build('layer_00',shared_inputs=args.shared_inputs,norm_bridge=args.norm_bridge)
    # The bank fixture was built against actual complete compile007 payloads.
    compiled={tuple(pe):p['parameters'].get('bank_words',0) for p in json.loads((reference/'source/profiles.json').read_text())['profiles'] for pe in p['pes']}
    if compiled!={tuple(p['pe']):p['parameters'].get('bank_words',0) for p in profiles}:raise ValueError('Original bank extents changed')
    workers=[]
    for p in profiles:
        if 'arm' not in p:continue
        v=p['parameters'];workers.append(dict(pe=p['pe'],role='gate_up' if v['branches']==2 else 'down',
            columns=v['columns'],max_parts=v['max_parts'],parts=p['arm']['parts'],iterations=p['arm']['iterations'],
            native_input_slices=p['native_input_slices']))
    files['workers.json']=(json.dumps(workers,separators=(',',':'))+'\n').encode()
    for n in ['fixture.json','bank-index.json','silu-proof.json']:files[n]=(reference/n).read_bytes()
    config=dict(norm_bridge=args.norm_bridge,complete_original_mlp=True,stage='layer_00',revision=meta['revision'],application=[width,height],
        application_pes=width*height,mlp_shape=meta['original_mlp_shape'],fabric_offset=[67,1],logical_origin=[63,0],
        artifact_single_message_limit=64<<20,compiler_timeout_seconds=900,run_timeout_seconds=900,
        fixture_metadata_sha256=expected,acceptance=meta['acceptance'],full_model=False,scope=meta['scope'])
    files['experiment.json']=(json.dumps(config,indent=2)+'\n').encode()
    for n in ['bounded_client.py','run_hw.py','job_capture.py','source_gate.py','elf_inventory.py']:
        files[n]=(ROOT/'runtime'/n).read_bytes()
    for n in ['backend.py','bounded_compiler.py','supervise.py','check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
    files['compile_hw.py']=(ROOT/'performance/runtime/compile_layer_mlp.py').read_bytes()
    files['run.py']=(ROOT/'performance/runtime/run_layer_mlp.py').read_bytes()
    for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
    session=['sh','/path/to/alcf-session.sh','host'];reuse=None
    if args.reuse_download:
        if not re.fullmatch('layer-mlp-hw-[0-9]{3}',args.reuse_download):raise ValueError('Reuse identity')
        old=ROOT/'performance/evidence'/args.reuse_download
        audit=json.loads((old/'compile-audit.json').read_text())
        if audit['exit_code']!=1 or audit['error'] is not None or audit['cleanup_errors'] or not audit['jobs']:
            raise ValueError('Unexpected original compiler service audit')
        if not all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in audit['jobs']):raise ValueError('Compiler service not released')
        if 'Image identity or size' not in (old/'compile.log').read_text():raise ValueError('Different original failure')
        previous='/srv/qwen38-singlewse-hardware/'+args.reuse_download
        script='root='+repr(previous)+'\n'+'''from pathlib import Path
import hashlib,json
r=Path(root);artifacts=list(r.glob('*.tar.gz'));assert len(artifacts)==1
p=artifacts[0];h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps(dict(artifact_name=p.name,artifact_sha256=h.hexdigest(),artifact_bytes=p.stat().st_size,
 csl_hashes={n:v for n,v in json.loads((r/'source-manifest.json').read_text())['files'].items() if n.endswith('.csl')})))
'''
        result=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
        reuse=json.loads(result.stdout);reuse['source_attempt']=args.reuse_download;reuse['original_host_admission_failure']='Image identity or size'
        if reuse['csl_hashes']!={n:hashlib.sha256(b).hexdigest() for n,b in files.items() if n.endswith('.csl')}:raise ValueError('Recompile required for changed CSL')
        files['reuse-compiler-output.json']=(json.dumps(reuse,indent=2)+'\n').encode()
        files['reused-compile-audit.json']=(old/'compile-audit.json').read_bytes()
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            entry=tarfile.TarInfo(n);entry.size=len(b);archive.addfile(entry,io.BytesIO(b))
    dest='/srv/qwen38-singlewse-hardware/'+name
    subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=payload.getvalue(),check=True,timeout=30)
    # Record the immutable source before payload transport so a transfer failure
    # cannot be confused with an absent or reusable attempt.
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'staging.json').write_text(json.dumps(dict(remote=dest,reference=args.reference,physical_dispatched=False),indent=2)+'\n')
    if reuse:
        admitted='/srv/qwen38-singlewse-hardware/'+args.reuse_admitted if args.reuse_admitted else None
        script='previous='+repr(previous)+'\nroot='+repr(dest)+'\ninfo='+repr(reuse)+'\nadmitted='+repr(admitted)+'\n'+'''import hashlib,json,os,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_FSIZE,(128<<20,128<<20));resource.setrlimit(resource.RLIMIT_CPU,(90,90))
r=Path(root);old=Path(previous);os.chdir(r)
from source_gate import verify
from backend import file_sha256
from compile_hw import admit
verify();meta=json.loads((r/'fixture.json').read_text())
for name in ['banks.npy','fixture.npz','silu-table.npy']:
 assert file_sha256(old/name)==meta['hashes'][name];os.link(old/name,r/name)
artifact=r/info['artifact_name'];assert file_sha256(old/artifact.name)==info['artifact_sha256']
os.link(old/artifact.name,artifact)
if admitted is None:result=admit(r,artifact,json.loads((r/'experiment.json').read_text()))
else:
 from check_sram import check
 source=Path(admitted);old_admission=json.loads((source/'artifact-admission.json').read_text())
 assert old_admission['passed'] and old_admission['artifact_sha256']==info['artifact_sha256']
 assert old_admission['sram_census_sha256']==file_sha256(source/'sram.json')
 assert {n:v for n,v in json.loads((source/'source-manifest.json').read_text())['files'].items() if n.endswith('.csl')}==info['csl_hashes']
 (r/'out/bin').mkdir(parents=True)
 for image in json.loads((source/'sram.json').read_text())['records']:
  path=Path(image['file']);assert not path.is_absolute() and '..' not in path.parts
  assert file_sha256(source/path)==image['sha256'];os.link(source/path,r/path)
 result=check(r,offset=(67,1));record=json.loads((source/'artifact.json').read_text())
 record['artifact']=str(artifact);(r/'artifact.json').write_text(json.dumps(record,indent=2)+'\\n')
verify()
receipt=dict(passed=result['passed'],physical_job_submitted=False,source_manifest_sha256=file_sha256(r/'source-manifest.json'),
 artifact_sha256=info['artifact_sha256'],application_pes=result['application_pes'],
 maximum_with_stack=max(x['low_section_end']+x['stack_allowance_bytes'] for x in result['records']),
 sram_census_sha256=file_sha256(r/'sram.json'),compiler_service_reused=info['source_attempt'],
 original_host_admission_failure=info['original_host_admission_failure'],individual_elf_file_limit=2<<20,
 images_reused_from=admitted)
(r/'artifact-admission.json').write_text(json.dumps(receipt,indent=2)+'\\n');print(json.dumps(receipt))
'''
        result=subprocess.run(session+['timeout --signal=TERM --kill-after=5 120 /opt/cerebras/venv/bin/python -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=135)
        (out/'artifact-admission.json').write_text(json.dumps(json.loads(result.stdout),indent=2)+'\n')
        (out/'payload-staging.json').write_text(json.dumps(dict(payload_verified=True,reused_remote_files=True,source_attempt=args.reuse_download,allocated_bank_bytes=meta['allocated_bank_bytes']),indent=2)+'\n')
        print(json.dumps(dict(name=name,compiler_service_reused=args.reuse_download,physical_dispatched=False)),flush=True)
        return
    if args.reuse_payload:
        previous='/srv/qwen38-singlewse-hardware/'+args.reuse_payload
        script='previous='+repr(previous)+'\nroot='+repr(dest)+'\n'+'''import json,os,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_CPU,(30,30));r=Path(root);old=Path(previous);os.chdir(r)
from source_gate import verify
from backend import file_sha256
verify();meta=json.loads((r/'fixture.json').read_text())
assert file_sha256(old/'fixture.json')==file_sha256(r/'fixture.json')
for name in ['banks.npy','fixture.npz','silu-table.npy']:
 assert file_sha256(old/name)==meta['hashes'][name];os.link(old/name,r/name)
verify();print(json.dumps(dict(payload_verified=True,reused_remote_files=True,payload_source=old.name,
 allocated_bank_bytes=meta['allocated_bank_bytes'],compiler_service_reused=False)))
'''
        result=subprocess.run(session+['timeout --signal=TERM --kill-after=5 45 /opt/cerebras/venv/bin/python -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=60)
        (out/'payload-staging.json').write_text(result.stdout)
        print(json.dumps(dict(name=name,payload_reused=args.reuse_payload,fresh_compilation_required=True,physical_dispatched=False)),flush=True)
        return
    remote=json.loads((reference/'dispatch.json').read_text())['remote']
    if not re.fullmatch('/srv/model-storage/qwen38-singlewse/runs/[a-zA-Z0-9_-]+',remote):raise ValueError('Frozen reference dispatch identity')
    reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(remote)+' banks.npy fixture.npz silu-table.npy'],stdout=subprocess.PIPE)
    try:subprocess.run(session+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=600)
    finally:
        reader.stdout.close()
        try:
            if reader.wait(timeout=20)!=0:raise RuntimeError('Payload source transport failed')
        except subprocess.TimeoutExpired:reader.kill();reader.wait();raise
    script='root='+repr(dest)+'\n'+'''import hashlib,json
from pathlib import Path
r=Path(root);meta=json.loads((r/'fixture.json').read_text())
for name,expected in meta['hashes'].items():
 h=hashlib.sha256()
 with (r/name).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==expected,name
print(json.dumps(dict(payload_verified=True,allocated_bank_bytes=meta['allocated_bank_bytes'])))
'''
    result=subprocess.run(session+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
    (out/'payload-staging.json').write_text(result.stdout)
    print(json.dumps(dict(name=name,source_files=len(files),native_workers=len(workers),physical_dispatched=False)),flush=True)


if __name__=='__main__':main()
