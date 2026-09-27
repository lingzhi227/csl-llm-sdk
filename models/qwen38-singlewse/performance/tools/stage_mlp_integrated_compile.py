"""Bounded compile admission of real MLP role bodies plus full static route audit.

No simulator execution, model payload transfer or physical allocation occurs.
The selected role census is explicit; it is not a full-wafer compiler admission.
"""
import argparse,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.contraction import tree
from spatial.sdk_leases import audit_explicit_sdk_leases
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--audit-from');a=p.parse_args()
if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
name='mlp-integrated-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
if out.exists():raise ValueError('Frozen attempt')
remote='/srv/model-storage/qwen38-singlewse/runs/'+name
active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
files={}
for dest,source in {
 'mlp_worker.csl':'performance/runtime/mlp_worker.csl','mlp_norm.csl':'performance/runtime/mlp_norm.csl','mlp_join.csl':'performance/runtime/mlp_join.csl','mlp_activation.csl':'performance/runtime/mlp_activation.csl',
 'fp8_shape.csl':'performance/probes/native_shapes/fp8_shape.csl','fp8_unpack_shift.csl':'performance/csl/fp8_unpack_shift.csl',
 'mlp_fused.csl':'performance/csl/mlp_fused.csl','fp8_encode.csl':'performance/csl/fp8_encode.csl','qwen_math.csl':'csl/qwen_math.csl',
 'source_gate.py':'runtime/source_gate.py','elf_inventory.py':'runtime/elf_inventory.py',
 'check_sram.py':'performance/runtime/check_sram.py','placement.py':'performance/runtime/placement.py'}.items():files[dest]=(ROOT/source).read_bytes()
paths=['configs/model-graph.json','configs/tensors.json']+['performance/'+s for s in ['spatial/mlp_static.py','spatial/mlp_regions.py','spatial/mlp_fusion.py','spatial/contraction.py','spatial/model_subgraph.py','tools/audit_mlp_static.py','tools/plan_mlp_static.py']]
for path in paths:files[path]=(ROOT/path).read_bytes()
nodes=tree(40);profiles=[]
for gate,fp8s in [(True,[129]*5),(False,[104,102,86,104,86])]:
 for (rank,pair),fp8_slots in zip([(0,0),(1,1),(39,63),(20,0),(2,63)],fp8s):
  n=nodes[rank];children=n['children']
  color=lambda node:4+2*(node['depth']-1)+node['side']
  params=dict(gate=gate,rank=rank,fp8_slots=fp8_slots,children=len(children),subtree=n['size'],
              left_color=color(nodes[children[0]]) if children else 4,
              right_color=color(nodes[children[1]]) if len(children)>1 else 5,
              native_out=color(n) if rank else 16 if gate else 17)
  profiles.append(dict(source='mlp_worker.csl',parameters=params))
for pair in [0,1,63]:profiles.append(dict(source='mlp_activation.csl',parameters=dict(pair=pair)))
for rank in [0,1,39]:profiles.append(dict(source='mlp_norm.csl',parameters=dict(rank=rank)))
for header in [False,True]:
 for parity in [0,1]:
  for has_child in [False,True]:
   profiles.append(dict(source='mlp_join.csl',parameters=dict(header=header,relay_only=False,has_child=has_child,
     own_color=13 if header else 2,child_color=17+(1-parity) if header else 13+(1-parity),out_color=17+parity if header else 13+parity)))
profiles.append(dict(source='mlp_join.csl',parameters=dict(header=True,relay_only=True,has_child=False,own_color=13,child_color=18,out_color=17)))
w=len(profiles)
layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={w},.height=1}});','layout {',f' @set_rectangle({w},1);']
for x,profile in enumerate(profiles):
 params=','.join('.%s=%s'%(k,str(v).lower() if isinstance(v,bool) else v) for k,v in profile['parameters'].items())
 layout.append(' @set_tile_code(%d,0,"%s",.{.memcpy_params=memcpy.get_params(%d),%s});'%(x,profile['source'],x,params))
layout+=[' @export_name("gains",[*]u16,false);',' @export_name("residual",[*]u16,false);',' @export_name("successor",[*]u16,false);',' @export_name("embedding_lookup",fn(u16,u16)void);',' @export_name("bank",[*]u32,false);',' @export_name("control",[*]u16,false);',' @export_name("native_local",[*]f32,false);',' @export_name("down_local",[*]f32,false);',' @export_name("value",[*]u32,false);',' @export_name("scale",[*]u32,false);',' @export_name("packet",[*]u32,false);',' @export_name("audit",[*]u32,false);',' @export_name("arm",fn(u16)void);',' @export_name("start",fn()void);','}']
files['layout.csl']=('\n'.join(layout)+'\n').encode()
files['profiles.json']=(json.dumps(dict(application=[w,1],profiles=profiles,full_wafer=False,executed=False),indent=2)+'\n').encode()
leases=audit_explicit_sdk_leases({k:v for k,v in files.items() if k.endswith('.csl') and k!='layout.csl' and b'@get_dsr' in v})
files['resource-audit.json']=(json.dumps(leases,indent=2)+'\n').encode()
if a.audit_from:
 if not (a.audit_from.startswith('mlp-integrated-compile-') and a.audit_from[-3:].isdigit()):raise ValueError('Audit source')
 old=ROOT/'performance/evidence'/a.audit_from/'static-routes.json'
 data=json.loads(old.read_text());assert data['audit']['passed']
 for n,h in data['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
 files['static-routes.json']=old.read_bytes()
files['execute.py']=f'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
commands=[['/usr/bin/python3','performance/tools/plan_mlp_static.py','--output','static-routes.json'],['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims={w+7},3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out']]
try:
 for phase,cmd in zip(['audit','compile'],commands):
  if phase=='audit' and {bool(a.audit_from)!r}:continue
  with Path(phase+'.log').open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    code=proc.wait(timeout=300)
    if code:raise RuntimeError(phase+' exit '+str(code))
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 result=check(Path.cwd(),application=({w},1),fabric=({w+7},3))
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,executed=False,application=[{w},1],sram_passed=result['passed'],seconds=time.monotonic()-started))+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''.encode()
files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
payload=io.BytesIO()
with tarfile.open(fileobj=payload,mode='w') as tar:
 for n,b in files.items():
  item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=620','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=30)
out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
print(json.dumps(dict(remote=remote,unit=unit,profiles=w,physical=False)))
