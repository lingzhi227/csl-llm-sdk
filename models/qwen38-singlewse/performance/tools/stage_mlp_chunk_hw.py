"""One bounded physical qualification of the existing resident helper protocol."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--simulation',required=True);p.add_argument('--reuse');a=p.parse_args()
if not re.fullmatch('[0-9]{3}',a.attempt) or not re.fullmatch('mlp-chunk-sim-[0-9]{3}',a.simulation):raise ValueError('Identity')
name='mlp-chunk-hw-'+a.attempt;out=ROOT/'evidence'/name
if out.exists():raise ValueError('Frozen attempt exists')
sim=ROOT/'performance/evidence'/a.simulation;receipt=json.loads((sim/'COMPLETE.json').read_text())
assert receipt['result']['passed'] and receipt['result']['normal_stop'] and not receipt['physical']
assert receipt['result']['all_original_banks_retained'] and receipt['result']['all_counters_exact']
assert receipt['result']['epochs']==3 and receipt['result']['partial_and_final_values_checked']==1152
assert json.loads((sim/'workstation-release.json').read_text())['workstation_released']
assert json.loads((sim/'source/resource-audit.json').read_text())['passed']
frozen=json.loads((sim/'source-manifest.json').read_text())['files'];files={}
for n,expected in frozen.items():
    if n.endswith('.csl') or n in ['prepare.py','resource-audit.json']:
        b=(sim/'source'/n).read_bytes();assert hashlib.sha256(b).hexdigest()==expected;files[n]=b
assert files['mlp_chunk_actor.csl']==(ROOT/'performance/protocols/mlp_chunk_actor.csl').read_bytes()
assert files['layout.csl']==(ROOT/'performance/probes/mlp_chunk/layout.csl').read_bytes()
# The numerical assertions and copies are unchanged. Only backend selection and
# the reported physical flag are adapted; do not repeat an unchanged simulation.
old=(sim/'source/run.py').read_text();assert hashlib.sha256(old.encode()).hexdigest()==frozen['run.py']
new=old.replace('import hashlib,json','import argparse,hashlib,json').replace('meta=json.loads',"parser=argparse.ArgumentParser();parser.add_argument('--physical',action='store_true');physical=parser.parse_args().physical\nmeta=json.loads",1).replace('with runtime(False)','with runtime(physical)').replace('normal_stop=True,physical=False','normal_stop=True,physical=physical')
start=new.index('parser=argparse.ArgumentParser()')
new=new[:start]+'def main():\n'+''.join(' '+line+'\n' if line else '\n' for line in new[start:].splitlines())+'\nif __name__=="__main__":main()\n'
assert new==(ROOT/'performance/probes/mlp_chunk/run.py').read_text();files['run.py']=new.encode()
files['driver-adaptation.json']=(json.dumps(dict(simulator_sha256=frozen['run.py'],physical_sha256=hashlib.sha256(new.encode()).hexdigest(),
    changes='Argparse --physical, backend argument, result physical flag and callable main with import guard for run_hw. CSL, fixture, neural/numeric assertions, counters and copies unchanged.'),indent=2)+'\n').encode()
files['fixture.json']=(sim/'fixture.json').read_bytes()
files['experiment.json']=(json.dumps(dict(application=[3,3],application_pes=9,artifact_single_message_limit=8<<20,
    full_model=False,compiler_timeout_seconds=600,run_timeout_seconds=300,simulation_protocol_source=a.simulation,compiled_component_reuse=bool(a.reuse),
    scope=receipt['result']['scope']),indent=2)+'\n').encode()
for n in ['bounded_client.py','run_hw.py','job_capture.py','source_gate.py','elf_inventory.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['backend.py','supervise.py','check_sram.py','placement.py']:
    files[n]=(ROOT/'performance/runtime'/n).read_bytes()
for n in ['check_sram.py','placement.py']:assert hashlib.sha256(files[n]).hexdigest()==frozen[n]
files['compile_hw.py']=(ROOT/'performance/runtime/compile_mlp_chunk.py').read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
dest='/srv/qwen38-singlewse-hardware/'+name;session=['sh','/path/to/alcf-session.sh','host']
reuse_copy=None
if a.reuse:
    if not re.fullmatch('mlp-chunk-hw-[0-9]{3}',a.reuse):raise ValueError('Reuse identity')
    previous=ROOT/'performance/evidence'/a.reuse
    ca=json.loads((previous/'compile-audit.json').read_text())
    assert ca['stage']=='compile' and ca['exit_code']==0 and not ca['cleanup_errors'] and ca['error'] is None
    assert ca['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in ca['jobs'])
    artifact=json.loads((previous/'artifact.json').read_text());sram=json.loads((previous/'sram.json').read_text())
    assert sram['passed'] and sram['application']==[3,3] and sram['application_pes']==9
    csl={n:hashlib.sha256(b).hexdigest() for n,b in files.items() if n.endswith('.csl')}
    old_manifest=json.loads((previous/'source-manifest.json').read_text())['files']
    assert all(old_manifest[n]==v for n,v in csl.items())
    elfs={Path(v['file']).name:v['sha256'] for v in sram['records']}
    reuse=dict(source_attempt=a.reuse,compiler_succeeded_and_released=True,artifact_sha256=artifact['sha256'],
               csl_hashes=csl,elf_hashes=elfs,application_pes=9)
    files['reuse-artifact.json']=(json.dumps(reuse,indent=2)+'\n').encode()
    files['reused-compile-audit.json']=(previous/'compile-audit.json').read_bytes()
    artifact_name=Path(artifact['artifact']).name;new_artifact=dict(artifact,artifact=dest+'/'+artifact_name)
    files['artifact.json']=(json.dumps(new_artifact,indent=2)+'\n').encode()
    reuse_copy=dict(previous='/srv/qwen38-singlewse-hardware/'+a.reuse,destination=dest,
                    artifact=artifact['artifact'],artifact_name=artifact_name,sha256=artifact['sha256'],elfs=elfs)
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tar:
    for n,b in files.items():
        member=tarfile.TarInfo(n);member.size=len(b);tar.addfile(member,io.BytesIO(b))
subprocess.run(session+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buffer.getvalue(),check=True,timeout=30)
if reuse_copy:
    script='info='+repr(reuse_copy)+'\n'+'''from pathlib import Path
import hashlib,shutil
src=Path(info['previous']);dst=Path(info['destination']);artifact=Path(info['artifact'])
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==info['sha256']
shutil.copyfile(artifact,dst/info['artifact_name']);(dst/'out/bin').mkdir(parents=True)
for n,h in info['elfs'].items():
 p=src/'out/bin'/n;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 shutil.copyfile(p,dst/'out/bin'/n)
'''
    subprocess.run(session+['python3 -c '+shlex.quote(script)],check=True,timeout=30)
source='/srv/model-storage/qwen38-singlewse/runs/'+a.simulation
reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(source)+' fixture.npz'],stdout=subprocess.PIPE)
try:subprocess.run(session+['tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=30)
finally:
    reader.stdout.close()
    try:assert reader.wait(timeout=20)==0
    except subprocess.TimeoutExpired:reader.kill();reader.wait();raise
expected=json.loads(files['fixture.json'])['fixture_sha256']
script='from pathlib import Path;import hashlib;p=Path('+repr(dest)+');assert hashlib.sha256((p/"fixture.npz").read_bytes()).hexdigest()=='+repr(expected)
subprocess.run(session+['python3 -c '+shlex.quote(script)],check=True,timeout=20)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
(out/'staging.json').write_text(json.dumps(dict(remote=dest,simulation=a.simulation,physical_dispatched=False),indent=2)+'\n')
print(json.dumps(dict(name=name,physical_dispatched=False)))
