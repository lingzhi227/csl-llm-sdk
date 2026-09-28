"""Bounded complete original-layer packing and independent MLP numerical oracle."""
import argparse,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--norm-bridge',action='store_true');parser.add_argument('--compiled',default='layer-mlp-compile-007');args=parser.parse_args()
if len(args.attempt)!=3 or not args.attempt.isdigit():raise ValueError('Attempt')
name='layer-mlp-reference-'+args.attempt;out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen attempt')
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
if active.stdout.strip():raise ValueError('Active workstation owner: '+active.stdout)
if not __import__('re').fullmatch('layer-mlp-compile-[0-9]{3}',args.compiled):raise ValueError('Compiled attempt identity')
compiled=ROOT/'performance/evidence'/args.compiled
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
binding=dict(compiled_profiles_sha256=hashlib.sha256(files['profiles.json']).hexdigest(),compile=args.compiled,scope='Complete original-layer bank initialization and full-MLP reference; no device/model execution.')
if args.norm_bridge:
    profiles=json.loads(files['profiles.json'])['profiles']
    if sum(len(p['pes']) for p in profiles if p['source']=='layer_norm_bridge.csl')!=40:raise ValueError('Admitted forty norm owners required')
    base=ROOT/'performance/evidence/layer-mlp-reference-001'
    original=json.loads((base/'source/profiles.json').read_text())['profiles']
    def banks(records):return {tuple(pe):r['parameters'].get('bank_words',0) for r in records for pe in r['pes']}
    remaps=json.loads((compiled/'source/state-page-remap.json').read_text())
    files['state-page-remap.json']=(json.dumps(remaps,indent=2)+'\n').encode()
    if sum(banks(profiles).values())!=sum(banks(original).values()):raise ValueError('Original resident capacity changed')
    binding.update(norm_bridge=True,bank_source='/srv/model-storage/qwen38-singlewse/runs/layer-mlp-reference-001',base_fixture_sha256=hashlib.sha256((base/'fixture.json').read_bytes()).hexdigest())
    for module_name in ['prepare_layer_norm','norm_oracle','remap_banks']:files[module_name+'.py']=(ROOT/'performance/reference'/f'{module_name}.py').read_bytes()
files['binding.json']=(json.dumps(binding,indent=2)+'\n').encode()
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
if args.norm_bridge:files['execute.py']=files['execute.py'].replace(b'from prepare_layer_mlp import main',b'from prepare_layer_norm import main')
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
