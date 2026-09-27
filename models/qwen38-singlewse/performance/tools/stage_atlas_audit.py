"""Bounded full-metadata audit on workstation; never allocates hardware."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('attempt')
args = parser.parse_args()
if len(args.attempt) != 3 or not args.attempt.isdigit():
    raise ValueError('Attempt must be three digits')
name = 'atlas-audit-' + args.attempt
remote = '/srv/model-storage/qwen38-singlewse/runs/' + name
unit = 'qwen38-single-' + name
out = ROOT / 'performance/evidence' / name
if out.exists():
    raise ValueError('Frozen local attempt exists')
active = subprocess.run(['ssh', 'workstation', 'systemctl --user list-units --type=service --state=running --no-legend qwen38-single-*'],
                        capture_output=True, text=True, check=True, timeout=20)
if active.stdout.strip():
    raise RuntimeError('Existing workstation owner: ' + active.stdout)
files = {('configs/' + n): (ROOT / 'configs' / n).read_bytes() for n in
         ['model-graph.json', 'tensors.json', 'config.json', 'rotary-frequencies.json']}
files['model-atlas.json'] = (ROOT / 'performance/evidence/model-atlas-001.json').read_bytes()
files['audit_atlas.py'] = (ROOT / 'performance/tools/audit_atlas.py').read_bytes()
files['atlas_builder.py'] = (ROOT / 'performance/spatial/atlas.py').read_bytes()
files['source_gate.py'] = (ROOT / 'runtime/source_gate.py').read_bytes()
files['execute.py'] = b'''import json,time
from pathlib import Path
from source_gate import verify
from audit_atlas import audit
started=time.monotonic()
try:
 verify();result=audit(Path('model-atlas.json'),Path('configs'));verify()
 Path('result.json').write_text(json.dumps(result,indent=2)+'\\n')
 Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-started,result=result),indent=2)+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''
files['source-manifest.json'] = (json.dumps(dict(files={n: hashlib.sha256(b).hexdigest() for n, b in files.items()}), indent=2) + '\n').encode()
size = sum(map(len, files.values()))
if size > 8 << 20:
    raise ValueError('Metadata transfer budget exceeded')
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w') as tar:
    for n, b in files.items():
        item = tarfile.TarInfo(n); item.size = len(b)
        tar.addfile(item, io.BytesIO(b))
subprocess.run(['ssh', 'workstation', 'mkdir ' + shlex.quote(remote) + ' && tar -xf - -C ' + shlex.quote(remote)],
               input=buffer.getvalue(), check=True, timeout=30)
command = ['systemd-run', '--user', '--unit=' + unit, '--property=MemoryMax=2G', '--property=MemorySwapMax=0',
           '--property=TasksMax=64', '--property=CPUQuota=200%', '--property=AllowedCPUs=6,7',
           '--property=RuntimeMaxSec=240', '--property=LimitFSIZE=16777216', '--property=LimitCORE=0',
           '--working-directory=' + remote, '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1',
           '--setenv=PYTHONDONTWRITEBYTECODE=1', '/usr/bin/taskset', '--cpu-list', '6,7', '/usr/bin/flock',
           '-n', '/srv/cerebras-workstation/heavy.lock', '/usr/bin/python3', 'execute.py']
subprocess.run(['ssh', 'workstation', shlex.join(command)], check=True, timeout=30)
out.mkdir()
(out / 'source-manifest.json').write_bytes(files['source-manifest.json'])
(out / 'dispatch.json').write_text(json.dumps(dict(remote=remote, unit=unit, physical=False, memory_max=2 << 30,
                                                 swap_max=0, runtime_max_seconds=240, metadata_bytes=size), indent=2) + '\n')
print(json.dumps(dict(remote=remote, unit=unit, metadata_bytes=size)))
