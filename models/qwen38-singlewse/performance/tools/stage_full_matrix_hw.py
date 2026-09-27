"""Stage the simulator-qualified complete BF16 matrix slice at its original WSE-3 coordinates."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(); parser.add_argument('attempt'); parser.add_argument('--simulation', required=True); parser.add_argument('--reuse-compiler-output'); parser.add_argument('--memcpy-channels',type=int,choices=[1,2],default=2); args = parser.parse_args()
if not re.fullmatch('[0-9]{3}', args.attempt) or not re.fullmatch('full-bf16-matrix-sim-[0-9]{3}', args.simulation):
    raise ValueError('Attempt identity')
if (ROOT / 'evidence' / ('full-bf16-matrix-hw-' + args.attempt)).exists():
    raise ValueError('Frozen hardware attempt exists')
if args.reuse_compiler_output and not re.fullmatch('full-bf16-matrix-hw-[0-9]{3}',args.reuse_compiler_output):
    raise ValueError('Original compile attempt identity')
sim = ROOT / 'performance/evidence' / args.simulation
assert not json.loads((sim/'source/region.json').read_text()).get('diagnostic_simprint',False),'Diagnostic printing is not hardware qualification'
receipt = json.loads((sim / 'COMPLETE.json').read_text())
assert receipt['result']['passed'] and receipt['result']['normal_stop'] and not receipt['physical']
assert receipt['result']['application'] == [631, 2]
assert receipt['result']['original_target_weights_retained'] and receipt['result']['all_loaded_weights_retained']
assert receipt['result']['all_native_dots_subtrees_gathers_and_outputs_exact'] and receipt['result']['all_packet_and_callback_counters_exact']
assert receipt['result']['complete_original_matrix_shape']==[48,5120]
payload_checks=receipt['result']['exact_device_payload_checks']
assert len(payload_checks)==2 and {v['kind'] for v in payload_checks}=={0,1}
for v in payload_checks:
    assert v['compared_words']==960*(65 if v['kind']==0 else 128)
    assert v['negative_control_mismatches']==1 and v['restored_mismatches']==0
assert json.loads((sim/'source/resource-audit.json').read_text())['passed']
assert json.loads((sim / 'workstation-release.json').read_text())['workstation_released']
frozen = json.loads((sim / 'source-manifest.json').read_text())['files']
files = {}
for n, expected in frozen.items():
    if n.endswith('.csl') or n in ['run.py', 'prepare.py', 'region.json', 'banks.json', 'weights.py', 'tensors.json', 'resource-audit.json']:
        data = (sim / 'source' / n).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected
        files[n] = data
files['experiment.json'] = (json.dumps(dict(application_pes=1262, application=[631, 2], matrix_shape=[48,5120], fabric_offset=[123,706], complete_original_matrix=True, full_model=False,
                                          memcpy_channels=args.memcpy_channels,simulation_memcpy_channels=2,
                                          artifact_single_message_limit=16 << 20, compiler_timeout_seconds=600,
                                          run_timeout_seconds=300, simulation_protocol_source=args.simulation,scope=receipt['result']['scope']), indent=2) + '\n').encode()
for n in ['bounded_client.py', 'run_hw.py', 'job_capture.py', 'source_gate.py', 'elf_inventory.py']:
    files[n] = (ROOT / 'runtime' / n).read_bytes()
for n in ['backend.py', 'supervise.py', 'check_sram.py', 'placement.py']:
    files[n] = (ROOT / 'performance/runtime' / n).read_bytes()
for n in ['check_sram.py', 'placement.py']:
    assert hashlib.sha256(files[n]).hexdigest() == frozen[n]
files['compile_hw.py'] = (ROOT / 'performance/runtime/compile_full_matrix_component.py').read_bytes()
for n in ['lifecycle.py', 'store.py', '__init__.py']:
    files['runtime/' + n] = (ROOT / 'runtime' / n).read_bytes()
if args.reuse_compiler_output:
    previous=ROOT/'performance/evidence'/args.reuse_compiler_output
    audit=json.loads((previous/'compile-audit.json').read_text())
    assert audit['exit_code']==1 and audit['error'] is None and not audit['cleanup_errors'] and not audit['uncorrelated_new_jobs']
    assert audit['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in audit['jobs'])
    old_manifest=json.loads((previous/'source-manifest.json').read_text())['files']
    assert json.loads((previous/'source/experiment.json').read_text()).get('memcpy_channels',2)==args.memcpy_channels,'Host transport change requires fresh compilation'
    csl_hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items() if n.endswith('.csl')}
    assert csl_hashes=={n:h for n,h in old_manifest.items() if n.endswith('.csl')}
    source_remote='/srv/qwen38-singlewse-hardware/'+args.reuse_compiler_output
    script='root_name='+repr(source_remote)+'\n'+'''from pathlib import Path
import json,hashlib
p=Path(root_name);a=list(p.glob('*.tar.gz'));assert len(a)==1 and 8<<20<a[0].stat().st_size<=16<<20
print(json.dumps(dict(artifact_file=a[0].name,artifact_bytes=a[0].stat().st_size,artifact_sha256=hashlib.sha256(a[0].read_bytes()).hexdigest())))
'''
    reply=subprocess.run(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(script)],check=True,capture_output=True,text=True,timeout=20)
    reused=dict(json.loads(reply.stdout),source_attempt=args.reuse_compiler_output,compiler_succeeded_and_released=True,original_host_stage_exit_code=1,csl_hashes=csl_hashes)
    files['reuse-compiler-output.json']=(json.dumps(reused,indent=2)+'\n').encode()
    files['reused-compile-audit.json']=(previous/'compile-audit.json').read_bytes()
    files['supervise.py']=(ROOT/'performance/runtime/supervise_reused_matrix.py').read_bytes()
    files['admit_reused_matrix.py']=(ROOT/'performance/runtime/admit_reused_matrix.py').read_bytes()
    files['verify_upload.py']=(ROOT/'performance/tools/verify_component_upload.py').read_bytes()
files['source-manifest.json'] = (json.dumps(dict(files={n: hashlib.sha256(b).hexdigest() for n, b in files.items()}), indent=2) + '\n').encode()
name = 'full-bf16-matrix-hw-' + args.attempt
destination = '/srv/qwen38-singlewse-hardware/' + name
session = ['sh', '/path/to/alcf-session.sh', 'host']
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w') as tar:
    for n, b in files.items():
        item = tarfile.TarInfo(n); item.size = len(b); tar.addfile(item, io.BytesIO(b))
subprocess.run(session + ['mkdir ' + shlex.quote(destination) + ' && tar -xf - -C ' + shlex.quote(destination)], input=buffer.getvalue(), check=True, timeout=30)
source_remote = '/srv/model-storage/qwen38-singlewse/runs/' + args.simulation
reader = subprocess.Popen(['ssh', 'workstation', 'tar -cf - -C ' + shlex.quote(source_remote) + ' fixture.npz fixture.json'], stdout=subprocess.PIPE)
try:
    subprocess.run(session + ['tar -xf - -C ' + shlex.quote(destination)], stdin=reader.stdout, check=True, timeout=30)
finally:
    reader.stdout.close()
    try:
        assert reader.wait(timeout=20) == 0
    except subprocess.TimeoutExpired:
        reader.kill(); reader.wait(); raise
expected = json.loads((sim / 'fixture.json').read_text())['fixture_sha256']
script = 'from pathlib import Path;import hashlib,json;p=Path(' + repr(destination) + ');m=json.loads((p/"fixture.json").read_text());assert hashlib.sha256((p/"fixture.npz").read_bytes()).hexdigest()==m["fixture_sha256"]==' + repr(expected)
subprocess.run(session + ['python3 -c ' + shlex.quote(script)], check=True, timeout=20)
if args.reuse_compiler_output:
    script='old='+repr('/srv/qwen38-singlewse-hardware/'+args.reuse_compiler_output)+'\nnew='+repr(destination)+'\n'+'''from pathlib import Path
import os,json,hashlib
p=Path(new);r=json.loads((p/'reuse-compiler-output.json').read_text());src=Path(old)/r['artifact_file']
assert src.stat().st_size==r['artifact_bytes'] and hashlib.sha256(src.read_bytes()).hexdigest()==r['artifact_sha256']
os.link(src,p/src.name)
'''
    subprocess.run(session+['python3 -c '+shlex.quote(script)],check=True,timeout=20)
    py='/opt/cerebras/venv/bin/python'
    for script_name in ['admit_reused_matrix.py','verify_upload.py']:
        subprocess.run(session+['cd '+shlex.quote(destination)+' && '+shlex.join([py,script_name])],check=True,timeout=60)
out = ROOT / 'evidence' / name; out.mkdir()
(out / 'source-manifest.json').write_bytes(files['source-manifest.json'])
(out / 'staging.json').write_text(json.dumps(dict(remote=destination, simulation=args.simulation, source='performance/probes/full_bf16_matrix', physical_dispatched=False), indent=2) + '\n')
print(json.dumps(dict(name=name, application_pes=1262, physical_dispatched=False)))
