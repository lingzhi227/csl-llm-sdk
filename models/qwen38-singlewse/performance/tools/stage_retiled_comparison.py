"""Bounded retained-fixture comparison after the candidate workstation run is terminal."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--candidate',required=True);a=p.parse_args()
assert re.fullmatch('[0-9]{3}',a.attempt) and re.fullmatch('retiled-matrix-sim-[0-9]{3}',a.candidate)
assert json.loads((ROOT/'performance/evidence'/a.candidate/'workstation-release.json').read_text())['workstation_released']
name='retiled-comparison-audit-'+a.attempt;remote='/srv/model-storage/qwen38-singlewse/runs/'+name
out=ROOT/'performance/evidence'/name;assert not out.exists()
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
assert not active.stdout.strip(),active.stdout
files={'compare_retiled_matrix.py':(ROOT/'performance/tools/compare_retiled_matrix.py').read_bytes(),'source_gate.py':(ROOT/'runtime/source_gate.py').read_bytes()}
files['execute.py']=('candidate='+repr(a.candidate)+'\n'+'''import json,time
from pathlib import Path
from source_gate import verify
from compare_retiled_matrix import compare
start=time.monotonic()
try:
 verify();runs=Path('/srv/model-storage/qwen38-singlewse/runs')
 result=compare(runs/'full-bf16-matrix-sim-017',runs/candidate);verify()
 Path('result.json').write_text(json.dumps(result,indent=2)+'\\n')
 Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-start,result=result),indent=2)+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
''').encode()
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as tar:
 for n,b in files.items():
  info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=buffer.getvalue(),check=True,timeout=30)
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=120','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=33554432','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=30)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,runtime_max_seconds=120,memory_max=2<<30,swap_max=0),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit)))
