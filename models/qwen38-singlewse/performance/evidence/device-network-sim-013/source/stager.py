"""Qualify actual full-bank cohosts, credited route turns and east-side SDK transit."""
import argparse,ast,hashlib,io,json,re,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'performance'),str(ROOT/'performance/tools')]
from stage_layer_mlp_compile import build
from spatial.device_network import lower_device_network
from spatial.layer_mlp_network import emit_routes
from spatial.layer_schedule import xy_rank


def main():
 p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--diagnostic',action='store_true');a=p.parse_args()
 if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
 name='device-network-sim-'+a.attempt;e=ROOT/'performance/evidence';out=e/name
 if out.exists():raise ValueError('Frozen attempt')
 active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
 if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
 files,_,_,profiles=build('layer_00',shared_inputs=True,norm_bridge=True,mixer_projections=True,joint_banks=True,routed_control=True)
 # Keep only actual compiler dependencies and explicit complete-bank identities.
 files={n:b for n,b in files.items() if n.endswith('.csl') or n in ['source_gate.py','elf_inventory.py','check_sram.py','placement.py','device_network.py','device_control.py','device-adaptation.json']}
 by_pe={tuple(p['pe']):p for p in profiles};old=e/'layer-mlp-compile-016/source'
 samples=json.loads((e/'layer-backend-compile-017/source/profiles.json').read_text())['samples'][1:]
 stage=json.loads((old/'joint-stage.json').read_text());region=next(r for r in stage['regions'] if r['role']=='mix')
 selected=[by_pe[tuple(s['original_pe'])] for s in samples];tags=[xy_rank(region,p['pe']) for p in selected]
 network=lower_device_network([0,0,3,3],tags);setups=dict((tuple(pe),s) for pe,s in json.loads((old/'mixer-setups.json').read_text()))
 fixture=e/'layer-joint-reference-001';receipt=json.loads((fixture/'bank-remap-proof.json').read_text());dispatch=json.loads((fixture/'dispatch.json').read_text())
 records=[]
 for profile,endpoint in zip(selected,network['workers']):
  params=dict(profile['parameters']);params.update(control_reply_rx=endpoint['reply_rx'],control_reply_tx=endpoint['reply_tx'])
  records.append(dict(pe=endpoint['pe'],original_pe=profile['pe'],tag=endpoint['tag'],source=profile['source'],parameters=params,setup=setups[tuple(profile['pe'])]))
 config=dict(workers=records,gateway=network['gateway'],source_root=dispatch['remote'],hashes=receipt['hashes'],complete_stage=False)
 files['selection.json']=(json.dumps(config,indent=2)+'\n').encode();files['device-network.json']=(json.dumps(network,indent=2)+'\n').encode()
 for n in ['sentinel.csl','prepare.py','run.py']:files[n]=(ROOT/'performance/probes/device_network'/n).read_bytes()
 files['backend.py']=(ROOT/'runtime/backend.py').read_bytes()
 if a.diagnostic:
  files['backend.py']=files['backend.py'].replace(b'suppress_trace=True',b'suppress_trace=False')
  files['run.py']=(ROOT/'performance/probes/device_network/diagnostic.py').read_bytes()
 controller=by_pe[tuple(stage['request_controller']['pe'])]
 layout=['const memcpy=@import_module("<memcpy/get_params>",.{.width=4,.height=3,.MEMCPYH2D_1=14});',
         'fn ordinary_memcpy_params(x:i16) memcpy.Params {var p=memcpy.get_params(x);p.MEMCPYH2D_1=-1;return p;}',
         'layout {',' @set_rectangle(4,3);']
 for record in records+[dict(pe=network['gateway'],source=controller['source'],parameters=controller['parameters'])]:
  x,y=record['pe'];params=','.join('.%s=%s'%(k,v if isinstance(v,str) else str(v).lower()) for k,v in record['parameters'].items())
  layout.append(f' @set_tile_code({x},{y},"{record["source"]}",.{{.memcpy_params=ordinary_memcpy_params({x}),{params}}});')
 for y in range(3):layout.append(f' @set_tile_code(3,{y},"sentinel.csl",.{{.memcpy_params=ordinary_memcpy_params(3)}});')
 layout.append(emit_routes(dict(rect=[0,0,4,3],routes=network['routes'])))
 exported={v.split(',')[-1].strip().strip('"') for v in re.findall(r'@export_symbol\(([^;]+)\);',files[controller['source']].decode())}
 declarations=[line for line in files['layout.csl'].decode().splitlines() if line.strip().startswith('@export_name')]
 layout+=[line for line in declarations if re.search(r'@export_name\("([^"]+)"',line).group(1) in exported]
 layout+=[' @export_name("sentinel",[*]u32,false);','}']
 files['layout.csl']=('\n'.join(layout)+'\n').encode()
 execute=(e/'device-control-sim-005/source/execute.py').read_bytes()
 execute=execute.replace(b'--fabric-dims=8,4',b'--fabric-dims=11,5').replace(b'application=(1,2),fabric=(8,4)',b'application=(4,3),fabric=(11,5)')
 execute=execute.replace(b'deadline=time.monotonic()+120',b"deadline=time.monotonic()+(600 if phase=='run' else 120)")
 ast.parse(execute);files['execute.py']=execute;files['stager.py']=Path(__file__).read_bytes()
 files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
 payload=io.BytesIO()
 with tarfile.open(fileobj=payload,mode='w') as archive:
  for n,b in files.items():
   item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
 remote='/srv/model-storage/qwen38-singlewse/runs/'+name
 subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
 unit='qwen38-single-'+name
 cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64',
      '--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=860','--property=TimeoutStopSec=5',
      '--property=KillMode=control-group','--property=LimitFSIZE=67108864','--property=LimitCORE=0','--working-directory='+remote,
      '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
      '/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
 subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
 out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
 (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,run_limit_seconds=600),indent=2)+'\n')
 print(json.dumps(dict(remote=remote,unit=unit,physical=False)))


if __name__=='__main__':main()
