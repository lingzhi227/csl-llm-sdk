"""Admit bounded 144-case norm validation after actual CSL compilation/SRAM.

The preceding simulator has partial numerical evidence, not complete acceptance.
The physical diagnostic must finish every case and stop/release normally.
"""
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='/srv/model-storage/gpt-oss20b'
SESSION=['sh','/path/to/alcf-session.sh','host']
NAME='norm-accuracy-hw-001'

def main():
    files={p.name:p.read_bytes() for p in (ROOT/'csl').glob('*.csl')}
    files.update({p.name:p.read_bytes() for p in (ROOT/'experiments/norm_accuracy').glob('*.csl')})
    hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    script='''from pathlib import Path
import hashlib,json
p=Path(BASE+'/runs/norm-accuracy-sim-001')
assert 'Compilation successful' in (p/'compile.log').read_text()
assert '"cases_completed": 24' in (p/'run.log').read_text()
s=json.loads((p/'sram.json').read_text());assert s['passed'] and s['application_pes']==2
assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in HASHES.items())
print(json.dumps(s))
'''.replace('BASE',repr(BASE)).replace('HASHES',repr(hashes))
    r=subprocess.run(['ssh','workstation','python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
    sram=json.loads(r.stdout)
    mapping={'run.py':'experiments/norm_accuracy/run.py','experiment.json':'experiments/norm_accuracy/experiment.json',
      'backend.py':'runtime/backend.py','compile_hw.py':'runtime/compile_micro_hw.py',
      'run_hw.py':'runtime/run_micro_hw.py','supervise.py':'runtime/supervise_micro_hw.py',
      'source_gate.py':'runtime/source_gate.py','job_capture.py':'runtime/job_capture.py',
      'check_sram.py':'tools/check_sram.py','elf_inventory.py':'tools/elf_inventory.py'}
    mapping.update({n:n for n in ('runtime/__init__.py','runtime/lifecycle.py','runtime/store.py')})
    files.update({n:(ROOT/path).read_bytes() for n,path in mapping.items()})
    admission=dict(scope='bounded physical RMSNorm precision diagnostic, 144 real inputs',
      full_simulator_acceptance=False,partial_simulator_cases_at_least=24,
      limitation='serial simulation is too slow for all 144 cases in its declared budget; no complete simulator claim',
      oracle_correction='one of 414720 initial FP64 oracle conversions double-rounded; new fixture quantizes directly on the BF16 lattice',
      physical_requirements='all 144 new/old comparisons, new <=1 BF16 ULP from direct FP64 rounding, all counts, normal stop and release',
      csl_hashes=hashes,sram=sram)
    files['admission.json']=(json.dumps(admission,indent=2)+'\n').encode()
    manifest={'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}}
    files['source-manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for name,data in files.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    dest='/srv/gpt-oss20b-hardware/'+NAME
    subprocess.run(SESSION+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=30)
    reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(BASE+'/fixtures/norm-accuracy-002')+' norm-fixture.npz norm-fixture.json'],stdout=subprocess.PIPE)
    try:subprocess.run(SESSION+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=60)
    finally:reader.stdout.close();assert reader.wait(timeout=15)==0
    subprocess.run(SESSION+['cd '+shlex.quote(dest)+" && /opt/cerebras/venv/bin/python -c 'from source_gate import verify; verify()'"],check=True,timeout=30)
    out=ROOT/'evidence'/NAME;out.mkdir(exist_ok=False)
    (out/'staging.json').write_text(json.dumps(dict(remote=dest,sources=manifest,admission=admission,submitted_job=False),indent=2)+'\n')
    print(json.dumps(dict(staged=True,remote=dest)))

if __name__=='__main__':main()
