"""Admit a fresh full-model diagnostic after full geometry compilation and SRAM."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from datetime import datetime,timezone
from pathlib import Path
from build_full_layout import build
ROOT=Path(__file__).resolve().parents[1]
BASE='/srv/model-storage/gpt-oss20b'
SESSION=['sh','/path/to/alcf-session.sh','host']

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--compile-attempt')
    p.add_argument('--cluster-compile-first',action='store_true')
    p.add_argument('--reuse-compiled',help='Reuse a released physical full-model compile with identical CSL')
    p.add_argument('--recover-artifact',help='Revalidate a successful compiler artifact whose old unpacked-size gate failed')
    a=p.parse_args()
    assert re.fullmatch(r'full-model-hw-[0-9]{3}',a.name)
    assert sum((bool(a.cluster_compile_first),a.compile_attempt is not None,a.reuse_compiled is not None,a.recover_artifact is not None))==1
    if a.compile_attempt is not None:assert re.fullmatch(r'[0-9]{3}',a.compile_attempt)
    if a.reuse_compiled:assert re.fullmatch(r'full-model-hw-[0-9]{3}',a.reuse_compiled) and a.reuse_compiled!=a.name
    if a.recover_artifact:assert re.fullmatch(r'full-model-hw-[0-9]{3}',a.recover_artifact) and a.recover_artifact!=a.name
    assert (ROOT/'experiments/full_model/layout.csl').read_text()==build()
    prerequisites=('moe-layer-hw-001','attention-prefix-hw-001','norm-accuracy-hw-001')
    for predecessor in prerequisites:
        accepted=json.loads((ROOT/'evidence'/predecessor/'COMPLETE.json').read_text());assert accepted['all_owned_jobs_released']
    norm_evidence=ROOT/'evidence/norm-accuracy-hw-001'
    norm_result=json.loads((norm_evidence/'result.json').read_text())
    assert norm_result['passed'] and norm_result['physical'] and norm_result['normal_stop'] and norm_result['cases']==144
    norm_manifest=json.loads((norm_evidence/'source-manifest.json').read_text())['files']
    assert norm_manifest['normalization_rope.csl']==hashlib.sha256((ROOT/'csl/normalization_rope.csl').read_bytes()).hexdigest()
    compiled=BASE+'/runs/full-model-compile-'+a.compile_attempt if a.compile_attempt else None
    files={f.name:f.read_bytes() for folder in ('csl','csl/resident') for f in (ROOT/folder).glob('*.csl')}
    files['layout.csl']=(ROOT/'experiments/full_model/layout.csl').read_bytes()
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    code='''import hashlib,json,pathlib,subprocess
p=pathlib.Path(COMPILED)
state=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','Result','-p','MainPID'],text=True)
assert 'ActiveState=inactive' in state and 'Result=success' in state and 'MainPID=0' in state,state
assert 'Compilation successful' in (p/'compile.log').read_text()
s=json.loads((p/'sram.json').read_text());assert s['passed'] and s['physical_application_pes']==870000
hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in NAMES}
print(json.dumps(dict(sram=s,hashes=hashes,state=state)))
'''.replace('COMPILED',repr(compiled)).replace('UNIT',repr('gptoss20b-full-model-compile-'+str(a.compile_attempt)+'.service')).replace('NAMES',repr(list(hashes)))
    reused=None;recovery=None
    if a.recover_artifact:
        previous='/srv/gpt-oss20b-hardware/'+a.recover_artifact
        recovery_code='''import hashlib,json,pathlib
p=pathlib.Path(PREVIOUS)
audit=json.loads((p/'compile-audit.json').read_text())
assert audit['exit_code']==1 and not audit['error'] and not audit['cleanup_errors'] and not audit['uncorrelated_new_jobs']
assert audit['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in audit['jobs'])
log=(p/'compile.log').read_text()
assert 'Compilation successful.' in log and 'ValueError: Full artifact exceeds archive bound' in log
manifest=json.loads((p/'source-manifest.json').read_text())['files']
assert {n:h for n,h in manifest.items() if n.endswith('.csl')}==HASHES
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in HASHES.items())
artifacts=list(p.glob('cs_*.tar.gz'));assert len(artifacts)==1
path=artifacts[0];assert path.stat().st_size<=512<<20
h=hashlib.sha256()
with path.open('rb') as f:
 for block in iter(lambda:f.read(4<<20),b''):h.update(block)
print(json.dumps(dict(source=str(p),artifact=str(path),sha256=h.hexdigest(),audit=audit,
 original_failure='successful compiler; old 512 MiB unpacked archive bound rejected its output')))
'''.replace('PREVIOUS',repr(previous)).replace('HASHES',repr(hashes))
        r=subprocess.run(SESSION+['python3 -c '+shlex.quote(recovery_code)],capture_output=True,text=True,check=True,timeout=30,cwd=ROOT)
        recovery=json.loads(r.stdout);predecessor={'sram':None}
    elif a.reuse_compiled:
        previous='/srv/gpt-oss20b-hardware/'+a.reuse_compiled
        reuse_code='''import hashlib,json,pathlib
p=pathlib.Path(PREVIOUS)
audit=json.loads((p/'compile-audit.json').read_text())
assert audit['exit_code']==0 and not audit['error'] and not audit['cleanup_errors'] and not audit['uncorrelated_new_jobs']
assert audit['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in audit['jobs'])
manifest=json.loads((p/'source-manifest.json').read_text())['files']
assert {n:h for n,h in manifest.items() if n.endswith('.csl')}==HASHES
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in HASHES.items())
sram=json.loads((p/'sram.json').read_text())
assert sram['passed'] and sram['physical_application_pes']==870000 and sram['records']
assert all(r['passed'] for r in sram['records'])
admission=json.loads((p/'PRE_RUN_ADMISSION.json').read_text())
assert all(admission[k] for k in ('passed','full_geometry_compiled','every_application_elf_sram_passed','embedded_sources_identical'))
assert admission['actual_unique_programs']==sram['unique_application_programs']
artifact=json.loads((p/'artifact.json').read_text());assert artifact['sha256']==admission['artifact_sha256']
path=pathlib.Path(artifact['artifact']);assert path.is_file() and path.stat().st_size<=512<<20
h=hashlib.sha256()
with path.open('rb') as f:
 for block in iter(lambda:f.read(4<<20),b''):h.update(block)
assert h.hexdigest()==artifact['sha256']
print(json.dumps(dict(source=str(p),audit=audit,artifact=artifact,sram=sram,admission=admission)))
'''.replace('PREVIOUS',repr(previous)).replace('HASHES',repr(hashes))
        r=subprocess.run(SESSION+['python3 -c '+shlex.quote(reuse_code)],capture_output=True,text=True,check=True,timeout=30,cwd=ROOT)
        reused=json.loads(r.stdout);predecessor={'sram':reused['sram']}
    elif a.cluster_compile_first:
        predecessor={'sram':None}
    else:
        r=subprocess.run(['ssh','workstation','python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=30,cwd=ROOT)
        predecessor=json.loads(r.stdout)
        if predecessor['hashes']!=hashes:raise ValueError('Full source differs from actual compiled/SRAM-audited programs')
    mapping={'run.py':'experiments/full_model/run.py','backend.py':'runtime/backend.py',
      'compile_hw.py':'runtime/compile_full_hw.py','run_hw.py':'runtime/run_micro_hw.py','supervise.py':'runtime/supervise_full_hw.py',
      'job_capture.py':'runtime/job_capture.py','source_gate.py':'runtime/source_gate.py',
      'check_sram.py':'tools/check_sram.py','elf_inventory.py':'tools/elf_inventory.py'}
    mapping.update({n:n for n in ('runtime/__init__.py','runtime/lifecycle.py','runtime/store.py')})
    mapping.update({n:n for n in ('core/checkpoint.py','core/resident_geometry.py','core/resident_identity.py','core/resident_loader.py','core/resident_schedule.py')})
    files.update({n:(ROOT/path).read_bytes() for n,path in mapping.items()})
    config=dict(full_model=True,shared_programs=True,physical_application_pes=870000,
      expected_unique_programs=None if a.cluster_compile_first or recovery else predecessor['sram']['unique_application_programs'],cluster_compile_first=a.cluster_compile_first,reuse_compiled=reused is not None,recover_artifact=recovery)
    files['experiment.json']=(json.dumps(config)+'\n').encode()
    admission=dict(full_geometry_compile=compiled,full_model_simulated=False,cluster_compile_first=a.cluster_compile_first,
      pre_allocation_gate='physical-fabric compilation, embedded source identity and every actual ELF SRAM must pass before runtime allocation',
      prerequisite_hardware=list(prerequisites),
      scope='bounded complete-original-model integration diagnostic; no full-model success claimed by staging',
      physical_requirements='all original weights once; two greedy tokens with persistent KV; post-token all-layer reference and all-PE counters; exhaustive weight retention; normal stop; owned-job release',
      bounds=dict(compile_seconds=2400,run_seconds=1800,client_address_space_bytes=8<<30,client_rss_bytes=2<<30,artifact_bytes=512<<20,unpacked_archive_bytes=2<<30,individual_elf_bytes=8<<20),
      source_hashes=hashes,sram=predecessor['sram'],reused_physical_compile=reused['source'] if reused else None,recovered_artifact=recovery)
    files['admission.json']=(json.dumps(admission,indent=2)+'\n').encode()
    if reused:
        for name,key in (('artifact.json','artifact'),('sram.json','sram'),('PRE_RUN_ADMISSION.json','admission')):
            files[name]=(json.dumps(reused[key],indent=2)+'\n').encode()
        files['reused-compile.json']=(json.dumps(dict(source=reused['source'],audit=reused['audit'],csl_hashes=hashes),indent=2)+'\n').encode()
    contract=dict(fixed_utc=datetime.now(timezone.utc).isoformat(),capture_observed=False,
      files={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
        ('docs/NUMERICAL-QUALIFICATION.md','reference/qualify_full_capture.py')})
    files['numerical-contract.json']=(json.dumps(contract,indent=2)+'\n').encode()
    manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
    files['source-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for name,data in files.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    dest='/srv/gpt-oss20b-hardware/'+a.name
    subprocess.run(SESSION+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=30,cwd=ROOT)
    reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(BASE+'/fixtures/full-model-001')+' full-fixture.npz full-fixture.json'],stdout=subprocess.PIPE,cwd=ROOT)
    try:subprocess.run(SESSION+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=60,cwd=ROOT)
    finally:
        reader.stdout.close()
        if reader.wait(timeout=15):raise RuntimeError('Reference fixture relay failed')
    verify="from source_gate import verify; verify(); import json; from pathlib import Path; r=json.loads(Path('/srv/gpt-oss20b-hardware/model/COMPLETE.json').read_text()); assert r['revision']=='6cee5e81ee83917806bbde320786a8fb61efebee' and r['tensors']==459"
    subprocess.run(SESSION+['cd '+shlex.quote(dest)+' && /opt/cerebras/venv/bin/python -c '+shlex.quote(verify)],check=True,timeout=30,cwd=ROOT)
    out=ROOT/'evidence'/a.name;out.mkdir(exist_ok=False)
    (out/'numerical-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (out/'staging.json').write_text(json.dumps(dict(remote=dest,sources=manifest,admission=admission,
      local_payload_files_created=False,submitted_hardware_job=False),indent=2)+'\n');print(json.dumps(dict(remote=dest,staged=True)))
    if recovery:
        inspect="from runtime.lifecycle import child_limits; child_limits('full-model'); from compile_hw import main; main()"
        subprocess.run(SESSION+['cd '+shlex.quote(dest)+' && /opt/cerebras/venv/bin/python -c '+shlex.quote(inspect)],check=True,timeout=180,cwd=ROOT)
        print(json.dumps(dict(remote=dest,existing_artifact_revalidated=True,new_compiler_job=False)))

if __name__=='__main__':main()
