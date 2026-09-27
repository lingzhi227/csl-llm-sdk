"""Bounded complete recurrent state and original resident FP8 cohost experiment."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'performance'))
from spatial.gdn_bank import GdnBankPlan
parser = argparse.ArgumentParser(); parser.add_argument('attempt'); args = parser.parse_args()
if len(args.attempt) != 3 or not args.attempt.isdigit():
    raise ValueError('Three-digit attempt required')
plan = GdnBankPlan(); plan.validate()
name = 'gdn-bank-sim-' + args.attempt
remote = '/srv/model-storage/qwen38-singlewse/runs/' + name
out = ROOT / 'performance/evidence' / name
if out.exists():
    raise ValueError('Frozen local attempt exists')
active = subprocess.run(['ssh', 'workstation', 'systemctl --user list-units --type=service --state=running --no-legend qwen38-single-*'],
                        capture_output=True, text=True, check=True, timeout=20)
if active.stdout.strip():
    raise ValueError('Workstation still owned: ' + active.stdout)
files = {n: (ROOT / 'performance/probes/gdn_bank' / n).read_bytes() for n in ['worker.csl', 'controller.csl', 'prepare.py', 'run.py']}
for n in ['gdn_shard32.csl', 'fp8_unpack.csl', 'fp8_dot.csl']:
    files[n] = (ROOT / 'performance/csl' / n).read_bytes()
for n in ['backend.py', 'source_gate.py', 'elf_inventory.py']:
    files[n] = (ROOT / 'runtime' / n).read_bytes()
for n in ['check_sram.py', 'placement.py']:
    files[n] = (ROOT / 'performance/runtime' / n).read_bytes()
files['reference_prepare.py'] = (ROOT / 'experiments/recurrent_head/prepare.py').read_bytes()
files['reference-metadata.json'] = (ROOT / 'evidence/recurrent-head-hw-003/fixture.json').read_bytes()
files['layout.csl'] = plan.emit_layout().encode()
files['region.json'] = (json.dumps(plan.document(), indent=2) + '\n').encode()
previous = ROOT / 'performance/evidence/mixed-bank-sim-002'
receipt = json.loads((previous / 'COMPLETE.json').read_text()); assert receipt['result']['passed'] and receipt['result']['normal_stop']
files['banks.json'] = (previous / 'fixture.json').read_bytes()
files['bank-source.json'] = (json.dumps(dict(source='mixed-bank-sim-002', fixture_sha256=json.loads(files['banks.json'])['fixture_sha256'],
                                          reuse='First111 original FP8 slots from each of six frozen banks, repeated over16 state PEs; representative atlas capacity, not all actual atlas owner payloads.'), indent=2) + '\n').encode()
files['REUSE.json'] = (json.dumps(dict(source_paths={'csl/recurrent_value64.csl': hashlib.sha256((ROOT / 'csl/recurrent_value64.csl').read_bytes()).hexdigest(),
                                                  'experiments/recurrent_head/prepare.py': hashlib.sha256(files['reference_prepare.py']).hexdigest()},
                                     note='Native FP32 recurrence vector order retained within each32-key shard; added four-block chain. Original FP64 propagated gate unchanged. Preprocessing remains outside this component.'), indent=2) + '\n').encode()
files['execute.py'] = b'''import subprocess,json,time,os,signal
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[['/usr/bin/python3','prepare.py'],[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=11,7','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],[sdk+'cs_python','run.py']]
try:
 for phase,cmd in zip(['prepare','compile','run'],commands):
  with Path(phase+'.log').open('x') as out:
   proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+240
   try:
    while proc.poll() is None:
     if time.monotonic()>deadline:raise TimeoutError(phase+' deadline')
     if phase=='run' and Path('sim.log').exists():
      if Path('sim.log').stat().st_size>8388608:raise RuntimeError('sim log budget')
      if 'FATAL:' in Path('sim.log').read_text():raise RuntimeError('Simulator fatal; frozen sim.log')
     time.sleep(.2)
    if proc.returncode:raise RuntimeError(phase+' exit '+str(proc.returncode))
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
  if phase=='compile':assert check(Path.cwd(),application=(4,5),fabric=(11,7))['application_pes']==20
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''
files['source-manifest.json'] = (json.dumps(dict(files={n: hashlib.sha256(b).hexdigest() for n, b in files.items()}), indent=2) + '\n').encode()
buffer = io.BytesIO()
with tarfile.open(fileobj=buffer, mode='w') as tar:
    for n, b in files.items():
        item = tarfile.TarInfo(n); item.size = len(b); tar.addfile(item, io.BytesIO(b))
subprocess.run(['ssh', 'workstation', 'mkdir ' + shlex.quote(remote) + ' && tar -xf - -C ' + shlex.quote(remote)], input=buffer.getvalue(), check=True, timeout=30)
previous_remote = '/srv/model-storage/qwen38-singlewse/runs/mixed-bank-sim-002/fixture.npz'
subprocess.run(['ssh', 'workstation', 'cp ' + shlex.quote(previous_remote) + ' ' + shlex.quote(remote + '/banks.npz')], check=True, timeout=30)
reference_cache = '/srv/model-storage/qwen38-singlewse/references/recurrent-head-hw-003'
for original, target in [('fixture.npz', 'reference.npz'), ('fixture.json', 'reference.json')]:
    subprocess.run(['ssh', 'workstation', 'cp --reflink=auto ' + shlex.quote(reference_cache + '/' + original) + ' ' + shlex.quote(remote + '/' + target)], check=True, timeout=30)
unit = 'qwen38-single-' + name
command = ['systemd-run', '--user', '--unit=' + unit, '--property=MemoryMax=2G', '--property=MemorySwapMax=0', '--property=TasksMax=64',
           '--property=CPUQuota=200%', '--property=AllowedCPUs=6,7', '--property=RuntimeMaxSec=760', '--property=LimitFSIZE=536870912', '--property=LimitCORE=0',
           '--working-directory=' + remote, '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=OMP_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
           '/usr/bin/taskset', '--cpu-list', '6,7', '/usr/bin/flock', '-n', '/srv/cerebras-workstation/heavy.lock', '/usr/bin/python3', 'execute.py']
subprocess.run(['ssh', 'workstation', shlex.join(command)], check=True, timeout=30)
out.mkdir(); (out / 'source-manifest.json').write_bytes(files['source-manifest.json'])
(out / 'dispatch.json').write_text(json.dumps(dict(remote=remote, unit=unit, physical=False, memory_max=2 << 30, swap_max=0, runtime_max_seconds=760), indent=2) + '\n')
print(json.dumps(dict(remote=remote, unit=unit)))
