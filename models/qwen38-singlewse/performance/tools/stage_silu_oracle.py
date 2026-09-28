"""One bounded exhaustive scalar-reference proof; no WSE allocation."""
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];name='layer-silu-oracle-001';out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen attempt')
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
if active.stdout.strip():raise ValueError('Live workstation owner')
files={'bf16_silu_lut.py':(ROOT/'performance/reference/bf16_silu_lut.py').read_bytes()}
files['execute.py']=b'''import json,sys,traceback
from pathlib import Path
from bf16_silu_lut import build_and_prove
import numpy as np
try:
 table,proof=build_and_prove()
 with Path('silu-table.npy').open('xb') as f:np.save(f,table)
 Path('COMPLETE.json').write_text(json.dumps(proof,indent=2)+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for n,b in files.items():i=tarfile.TarInfo(n);i.size=len(b);archive.addfile(i,io.BytesIO(b))
remote='/srv/model-storage/qwen38-singlewse/runs/'+name
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=buffer.getvalue(),check=True,timeout=20)
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=100%','--property=AllowedCPUs=6','--property=RuntimeMaxSec=60','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=8388608','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit,physical=False)))
