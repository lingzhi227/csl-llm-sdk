"""Admit one full expert from matching mini protocol and full SRAM evidence."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='/srv/model-storage/gpt-oss20b/runs/'
SESSION=['sh','/path/to/alcf-session.sh','host']

def remote_json(code):
    r=subprocess.run(['ssh','workstation','python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=30)
    return json.loads(r.stdout)

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--mini-attempt',required=True);p.add_argument('--full-attempt',required=True)
    p.add_argument('--allow-readback-timeout',action='store_true',
                   help='Explicitly admit two completed numerical/protocol epochs after a simulator-only final-readback deadline; physical readback/stop gates remain mandatory')
    a=p.parse_args()
    if not re.fullmatch(r'expert-full-hw-[0-9]{3}',a.name):raise ValueError('Fresh named expert attempt required')
    if not all(re.fullmatch(r'[0-9]{3}',v) for v in (a.mini_attempt,a.full_attempt)):raise ValueError('Invalid predecessor')
    mini=BASE+'expert-mini-sim-'+a.mini_attempt;full=BASE+'expert-full-sim-'+a.full_attempt
    mapping={n:'experiments/expert/'+n for n in ('layout.csl','pe.csl','run.py')}
    mapping.update({n:'csl/'+n for n in ('mxfp4_gemv.csl','moe_math.csl')})
    mapping.update({'backend.py':'runtime/backend.py','compile_hw.py':'runtime/compile_micro_hw.py',
                    'run_hw.py':'runtime/run_micro_hw.py','supervise.py':'runtime/supervise_micro_hw.py',
                    'job_capture.py':'runtime/job_capture.py','source_gate.py':'runtime/source_gate.py'})
    mapping.update({n:n for n in ('runtime/__init__.py','runtime/lifecycle.py','runtime/store.py')})
    mapping.update({n:'tools/'+n for n in ('check_sram.py','elf_inventory.py')})
    files={n:(ROOT/source).read_bytes() for n,source in mapping.items()}
    checked=[n for n in files if n.endswith('.csl') or n in ('run.py','backend.py')]
    sim_complete=True
    if a.allow_readback_timeout:
        prefix="import json,pathlib,subprocess; p=pathlib.Path("+repr(mini)+"); "
        prefix+="rows=json.loads((p/'observations.json').read_text()); assert len(rows)==2 and [r['epoch'] for r in rows]==[1,2]; "
        prefix+="assert all(r['every_pe_completed'] and all(c['passed'] for c in r['checks'].values()) for r in rows); "
        prefix+="unit='gptoss20b-expert-mini-run-"+a.mini_attempt+".service'; state=subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','-p','Result','-p','MainPID'],text=True); "
        prefix+="assert 'ActiveState=failed' in state and 'Result=timeout' in state and 'MainPID=0' in state; assert not (p/'result.json').exists(); "
        prefix+="print(json.dumps({'simulator_complete':False,'numerical_and_protocol_epochs':2,'final_weight_readback_completed':False,'unit_state':state}))"
        simulator_receipt=remote_json(prefix);sim_complete=False
    else:
        simulator_receipt=remote_json("import json,pathlib; p=pathlib.Path("+repr(mini)+"); r=json.loads((p/'result.json').read_text()); assert r['passed'] and r['normal_stop']; print(json.dumps({'simulator_complete':True}))")
    code="import json,pathlib,hashlib; m=pathlib.Path("+repr(mini)+"); f=pathlib.Path("+repr(full)+"); "
    code+="s=json.loads((f/'sram.json').read_text()); assert s['passed'] and s['application_pes']==828; "
    code+="print(json.dumps({str(p):{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in "+repr(checked)+"} for p in [m,f]}))"
    hashes=remote_json(code);expected={n:hashlib.sha256(files[n]).hexdigest() for n in checked}
    if any(v!=expected for v in hashes.values()):raise ValueError('Sources differ from accepted mini/full geometry')
    files['experiment.json']=(json.dumps(dict(application_pes=828,full_expert=True,compiler_params='gate_rows:36,down_rows:18'))+'\n').encode()
    admission=dict(mini_numerical_sim=mini,full_compile_and_sram=full,full_expert_simulated=False,
                   simulator=simulator_receipt,source_hashes=expected,
                   physical_requirements='two full-expert numerical epochs, every PE completed, exhaustive original weights/scales/bias readback, normal SDK stop and owned-job release')
    files['admission.json']=(json.dumps(admission,indent=2)+'\n').encode()
    manifest={'files':{n:hashlib.sha256(raw).hexdigest() for n,raw in files.items()}}
    files['source-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for name,raw in files.items():
            entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mode=0o600;archive.addfile(entry,io.BytesIO(raw))
    dest='/srv/gpt-oss20b-hardware/'+a.name
    subprocess.run(SESSION+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=30)
    reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(full)+' expert-fixture.npz expert-fixture.json'],stdout=subprocess.PIPE)
    try:subprocess.run(SESSION+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=60)
    finally:
        reader.stdout.close()
        if reader.wait(timeout=15):raise RuntimeError('Fixture relay failed')
    subprocess.run(SESSION+['cd '+shlex.quote(dest)+" && /opt/cerebras/venv/bin/python -c 'from source_gate import verify; verify()'"],check=True,timeout=30)
    receipt=dict(remote=dest,sources=manifest,admission=admission,fixture_relayed_without_local_payload_file=True,submitted_hardware_job=False)
    out=ROOT/'evidence'/a.name;out.mkdir(exist_ok=False)
    (out/'staging.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(remote=dest,staged=True)))

if __name__=='__main__':main()
