"""Stage one bounded whole-layer physical diagnostic from full geometry/SRAM evidence."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='/srv/model-storage/gpt-oss20b'
SESSION=['sh','/path/to/alcf-session.sh','host']

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--compile-attempt',required=True)
    p.add_argument('--kind',choices=('moe','attention','decoder'),default='moe')
    p.add_argument('--reuse-compiled',help='An earlier identical-CSL physical compile with successful released compiler job')
    a=p.parse_args()
    stem,folder,fixture_stem,physical_pes={'moe':('moe-layer','moe_layer','moe',31320),
      'attention':('attention-prefix','attention_prefix','attention',4640),
      'decoder':('decoder-layer','decoder_layer','full',31320)}[a.kind]
    assert re.fullmatch(stem+r'-hw-[0-9]{3}',a.name) and re.fullmatch(r'[0-9]{3}',a.compile_attempt)
    compiled=BASE+'/runs/'+stem+'-compile-'+a.compile_attempt
    files={f.name:f.read_bytes() for f in (ROOT/'csl').glob('*.csl')}
    roles={'moe':('expert.csl','expert_join.csl','idle.csl','moe_controller.csl','router.csl'),
      'attention':('projection.csl','prefix_collector.csl','kv.csl','attention_controller.csl'),
      'decoder':('projection_full.csl','prefix_collector.csl','kv.csl','attention_controller.csl','moe_controller_full.csl','router.csl','expert.csl','expert_join_full.csl','idle.csl')}[a.kind]
    files.update({n:(ROOT/'csl/resident'/n).read_bytes() for n in roles})
    files.update({f.name:f.read_bytes() for f in (ROOT/'experiments'/folder).glob('*.csl')})
    csl_hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    code='''import hashlib,json,pathlib,subprocess
p=pathlib.Path(COMPILED)
state=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','Result','-p','MainPID'],text=True)
assert 'ActiveState=inactive' in state and 'Result=success' in state and 'MainPID=0' in state,state
assert 'Compilation successful' in (p/'compile.log').read_text()
s=json.loads((p/'sram.json').read_text());assert s['passed'] and s['physical_application_pes']==PE_COUNT
hashes={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in NAMES}
print(json.dumps(dict(sram=s,hashes=hashes,state=state)))
'''.replace('COMPILED',repr(compiled)).replace('UNIT',repr('gptoss20b-'+stem+'-compile-'+a.compile_attempt+'.service')).replace('NAMES',repr(list(csl_hashes))).replace('PE_COUNT',str(physical_pes))
    result=subprocess.run(['ssh','workstation','python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=30)
    predecessor=json.loads(result.stdout)
    if predecessor['hashes']!=csl_hashes:raise ValueError('CSL differs from compiled/SRAM-audited geometry')
    mapping={'run.py':'experiments/'+folder+'/run.py','backend.py':'runtime/backend.py',
      'compile_hw.py':'runtime/compile_moe_hw.py','run_hw.py':'runtime/run_micro_hw.py','supervise.py':'runtime/supervise_moe_hw.py',
      'job_capture.py':'runtime/job_capture.py','source_gate.py':'runtime/source_gate.py',
      'check_sram.py':'tools/check_sram.py','elf_inventory.py':'tools/elf_inventory.py'}
    mapping.update({n:n for n in ('runtime/__init__.py','runtime/lifecycle.py','runtime/store.py')})
    if a.kind=='decoder':mapping.update({n:n for n in ('core/checkpoint.py','core/resident_geometry.py','core/resident_loader.py')})
    files.update({n:(ROOT/f).read_bytes() for n,f in mapping.items()})
    files['experiment.json']=(json.dumps(dict(shared_programs=True,physical_application_pes=physical_pes,full_moe_layer=a.kind in ('moe','decoder'),full_attention=a.kind in ('attention','decoder'),full_decoder_layer=a.kind=='decoder',
      expected_unique_programs=predecessor['sram']['unique_application_programs']))+'\n').encode()
    admission=dict(full_geometry_compile=compiled,full_layer_simulated=False,
      prerequisite_hardware=['router-hw-001','expert-full-hw-001'] if a.kind=='moe' else ['transformer-math-hw-001','router-hw-001'] if a.kind=='attention' else ['attention-prefix-hw-001','moe-layer-hw-001'],
      scope='bounded integration diagnostic: '+stem,
      physical_requirements='two complete real-input outputs, every PE epoch, exhaustive resident weight retention, normal stop and owned-job release; dynamic top4 for MoE or persistent KV for attention',
      bounds=dict(compile_seconds=900,run_seconds=900,client_address_space_bytes=4<<30,client_rss_bytes=1<<30),
      source_hashes=csl_hashes,sram=predecessor['sram'])
    files['admission.json']=(json.dumps(admission,indent=2)+'\n').encode()
    if a.reuse_compiled:
        assert re.fullmatch(stem+r'-hw-[0-9]{3}',a.reuse_compiled) and a.reuse_compiled!=a.name
        previous='/srv/gpt-oss20b-hardware/'+a.reuse_compiled
        code='''import json,pathlib
p=pathlib.Path(PREVIOUS)
audit=json.loads((p/'compile-audit.json').read_text());assert audit['exit_code']==0 and not audit['error'] and not audit['cleanup_errors']
assert audit['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in audit['jobs'])
manifest=json.loads((p/'source-manifest.json').read_text())['files']
assert all(manifest[n]==h for n,h in HASHES.items())
sram=json.loads((p/'sram.json').read_text());assert sram['passed'] and sram['physical_application_pes']==PE_COUNT
print(json.dumps(dict(artifact=(p/'artifact.json').read_text(),sram=(p/'sram.json').read_text(),audit=audit,source=str(p))))
'''.replace('PREVIOUS',repr(previous)).replace('HASHES',repr(csl_hashes)).replace('PE_COUNT',str(physical_pes))
        result=subprocess.run(SESSION+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=30,cwd=ROOT)
        reused=json.loads(result.stdout)
        files['artifact.json']=reused['artifact'].encode();files['sram.json']=reused['sram'].encode()
        files['reused-compile.json']=(json.dumps(dict(source=reused['source'],audit=reused['audit'],csl_hashes=csl_hashes),indent=2)+'\n').encode()
        config=json.loads(files['experiment.json']);config['reuse_compiled']=True
        files['experiment.json']=(json.dumps(config)+'\n').encode()
    manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
    files['source-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for name,data in files.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    dest='/srv/gpt-oss20b-hardware/'+a.name
    subprocess.run(SESSION+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=30,cwd=ROOT)
    fixture_names=' '+fixture_stem+'-fixture.npz '+fixture_stem+'-fixture.json'
    fixture_dir='full-model' if a.kind=='decoder' else stem
    reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(BASE+'/fixtures/'+fixture_dir+'-001')+fixture_names],stdout=subprocess.PIPE,cwd=ROOT)
    try:subprocess.run(SESSION+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=600,cwd=ROOT)
    finally:
        reader.stdout.close()
        try:status=reader.wait(timeout=15)
        except subprocess.TimeoutExpired:reader.terminate();reader.wait(timeout=10);raise
        if status:raise RuntimeError('Whole-layer fixture stream failed')
    subprocess.run(SESSION+['cd '+shlex.quote(dest)+" && /opt/cerebras/venv/bin/python -c 'from source_gate import verify; verify()'"],check=True,timeout=30,cwd=ROOT)
    out=ROOT/'evidence'/a.name;out.mkdir(exist_ok=False)
    (out/'staging.json').write_text(json.dumps(dict(remote=dest,sources=manifest,admission=admission,
      local_payload_files_created=False,submitted_hardware_job=False),indent=2)+'\n')
    print(json.dumps(dict(remote=dest,staged=True)),flush=True)

if __name__=='__main__':main()
