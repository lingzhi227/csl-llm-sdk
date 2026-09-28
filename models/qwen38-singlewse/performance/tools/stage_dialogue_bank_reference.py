"""Specialize the P39 bank fixture only after the full candidate passes SRAM."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    parser.add_argument('--compile', required=True)
    args = parser.parse_args()
    if any(len(v) != 3 or not v.isdigit() for v in (args.attempt, args.compile)):
        raise ValueError('Attempt identifier')
    name = 'layer-dialogue-reference-'+args.attempt
    evidence = ROOT/'performance/evidence'
    out = evidence/name
    compiled = evidence/('layer-mlp-compile-'+args.compile)
    if out.exists(): raise ValueError('Frozen attempt')
    if not json.loads((compiled/'COMPLETE.json').read_text())['sram_passed']:
        raise ValueError('Complete candidate has not passed SRAM admission')
    live = subprocess.run(['ssh', 'workstation', 'systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],
                          capture_output=True, text=True, check=True, timeout=20)
    if live.stdout.strip(): raise ValueError('Active workstation owner: '+live.stdout)
    files = {n: (compiled/'source'/n).read_bytes() for n in
             ['profiles.json', 'joint-bank-placement.json', 'joint-stage.json', 'dialogue-bank-placement.json', 'bank-specialization.json']}
    files['compiled-source-manifest.json'] = (compiled/'source-manifest.json').read_bytes()
    original = next(s for s in json.loads((evidence/'layer-native-schedule-002/layer-schedule.json').read_text())['stages'] if s['id'] == 'layer_00')
    files['original-stage.json'] = (json.dumps(original, separators=(',', ':'))+'\n').encode()
    baseline = evidence/'layer-joint-reference-003'
    proof = json.loads((baseline/'bank-remap-proof.json').read_text())
    files['source-bank.json'] = (json.dumps(dict(root=json.loads((baseline/'dispatch.json').read_text())['remote'],
                hashes=proof['hashes'], source_proof_sha256=hashlib.sha256((baseline/'bank-remap-proof.json').read_bytes()).hexdigest()), indent=2)+'\n').encode()
    files['prepare.py'] = (ROOT/'performance/reference/prepare_dialogue_banks.py').read_bytes()
    files['spatial/__init__.py'] = b''
    for module in ['layer_schedule', 'joint_placement', 'dialogue_placement']:
        files['spatial/'+module+'.py'] = (ROOT/'performance/spatial'/(module+'.py')).read_bytes()
    files['source_gate.py'] = (ROOT/'runtime/source_gate.py').read_bytes()
    files['stager.py'] = Path(__file__).read_bytes()
    files['execute.py'] = b'''import json,time
from pathlib import Path
from source_gate import verify
from prepare import main
started=time.monotonic()
try:
 verify();main();verify()
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,seconds=time.monotonic()-started,proof=json.loads(Path('bank-remap-proof.json').read_text())))+'\\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\\n');raise
'''
    files['source-manifest.json'] = (json.dumps(dict(files={n: hashlib.sha256(b).hexdigest() for n, b in files.items()}), indent=2)+'\n').encode()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w') as archive:
        for n, b in files.items():
            item = tarfile.TarInfo(n); item.size = len(b)
            archive.addfile(item, io.BytesIO(b))
    remote = '/srv/model-storage/qwen38-singlewse/runs/'+name
    subprocess.run(['ssh', 'workstation', 'mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)], input=payload.getvalue(), check=True, timeout=30)
    unit = 'qwen38-single-'+name
    cmd = ['systemd-run', '--user', '--unit='+unit, '--property=MemoryMax=2G', '--property=MemorySwapMax=0',
           '--property=TasksMax=32', '--property=CPUQuota=100%', '--property=AllowedCPUs=6', '--property=RuntimeMaxSec=180',
           '--property=TimeoutStopSec=5', '--property=KillMode=control-group', '--property=LimitFSIZE=536870912', '--property=LimitCORE=0',
           '--working-directory='+remote, '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
           '/usr/bin/taskset', '--cpu-list', '6', '/usr/bin/flock', '-n', '/srv/cerebras-workstation/heavy.lock', '/usr/bin/python3', 'execute.py']
    subprocess.run(['ssh', 'workstation', shlex.join(cmd)], check=True, timeout=20)
    out.mkdir()
    (out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote, unit=unit, physical=False, compile_attempt=compiled.name), indent=2)+'\n')
    print(json.dumps(dict(remote=remote, unit=unit, physical=False)))


if __name__ == '__main__': main()
