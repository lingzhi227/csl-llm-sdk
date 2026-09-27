"""Stage a qualified8x32 complete BF16 matrix at reference physical coordinates."""
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
parser = argparse.ArgumentParser(); parser.add_argument('attempt'); parser.add_argument('--simulation', required=True); parser.add_argument('--memcpy-channels',type=int,choices=[1,2],default=1); args = parser.parse_args()
if not re.fullmatch('[0-9]{3}', args.attempt) or not re.fullmatch('retiled-matrix-sim-[0-9]{3}', args.simulation):
    raise ValueError('Attempt identity')
if (ROOT / 'evidence' / ('retiled-matrix-hw-' + args.attempt)).exists():
    raise ValueError('Frozen hardware attempt exists')
sim = ROOT / 'performance/evidence' / args.simulation
assert not json.loads((sim/'source/region.json').read_text()).get('diagnostic_simprint',False),'Diagnostic printing is not hardware qualification'
receipt = json.loads((sim / 'COMPLETE.json').read_text())
assert receipt['result']['passed'] and receipt['result']['normal_stop'] and not receipt['physical']
assert receipt['result']['application'] == [631, 2]
assert receipt['result']['original_target_weights_retained'] and receipt['result']['all_loaded_weights_retained']
assert receipt['result']['all_native_dots_subtrees_gathers_and_outputs_exact'] and receipt['result']['all_packet_and_callback_counters_exact']
assert receipt['result']['complete_original_matrix_shape']==[48,5120]
assert json.loads((sim/'source/region.json').read_text())['matrix']['tile_shape']==[8,32]
payload_checks=receipt['result']['exact_device_payload_checks']
assert len(payload_checks)==2 and {v['kind'] for v in payload_checks}=={0,1}
for v in payload_checks:
    assert v['compared_words']==960*(17 if v['kind']==0 else 128)
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
files['source-manifest.json'] = (json.dumps(dict(files={n: hashlib.sha256(b).hexdigest() for n, b in files.items()}), indent=2) + '\n').encode()
name = 'retiled-matrix-hw-' + args.attempt
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
out = ROOT / 'evidence' / name; out.mkdir()
(out / 'source-manifest.json').write_bytes(files['source-manifest.json'])
(out / 'staging.json').write_text(json.dumps(dict(remote=destination, simulation=args.simulation, source='performance/probes/retiled_matrix', physical_dispatched=False), indent=2) + '\n')
print(json.dumps(dict(name=name, application_pes=1262, physical_dispatched=False)))
