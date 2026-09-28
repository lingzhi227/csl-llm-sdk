"""Bounded complete original-layer packing and independent MLP numerical oracle."""
import argparse,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
if len(args.attempt)!=3 or not args.attempt.isdigit():raise ValueError('Attempt')
name='layer-mlp-reference-'+args.attempt;out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen attempt')
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
if active.stdout.strip():raise ValueError('Active workstation owner: '+active.stdout)
compiled=ROOT/'performance/evidence/layer-mlp-compile-007'
assert json.loads((compiled/'COMPLETE.json').read_text())['passed']
assert json.loads((compiled/'workstation-release.json').read_text())['workstation_released']
plan=json.loads((ROOT/'performance/evidence/layer-native-schedule-002/layer-schedule.json').read_text())
stage=next(s for s in plan['stages'] if s['id']=='layer_00')
files={}
for dest,source in {'prepare_layer_mlp.py':'performance/reference/prepare_layer_mlp.py','mlp_oracle.py':'performance/reference/mlp_oracle.py',
                    'bf16_silu_lut.py':'performance/reference/bf16_silu_lut.py','layer_schedule.py':'performance/spatial/layer_schedule.py',
                    'weights.py':'runtime/weights.py','source_gate.py':'runtime/source_gate.py','tensors.json':'configs/tensors.json','hub.json':'configs/hub.json'}.items():files[dest]=(ROOT/source).read_bytes()
files['stage.json']=(json.dumps(stage,separators=(',',':'))+'\n').encode()
files['profiles.json']=(compiled/'source/profiles.json').read_bytes()
files['binding.json']=(json.dumps(dict(compiled_profiles_sha256=hashlib.sha256(files['profiles.json']).hexdigest(),compile='layer-mlp-compile-007',scope='Complete original-layer bank initialization and full-MLP reference; no device/model execution.'),indent=2)+'\n').encode()
files['execute.py']=b'''import json
from pathlib import Path
from source_gate import verify
try:
 verify()
 from prepare_layer_mlp import main
 main();verify()
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for n,b in files.items():i=tarfile.TarInfo(n);i.size=len(b);archive.addfile(i,io.BytesIO(b))
remote='/srv/model-storage/qwen38-singlewse/runs/'+name
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=buffer.getvalue(),check=True,timeout=30)
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=900','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,'--property=StandardOutput=append:'+remote+'/run.log','--property=StandardError=append:'+remote+'/run.log','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit,physical=False)))
