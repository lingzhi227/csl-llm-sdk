"""Bounded gateway/internal-PE protocol test using one unchanged original bank."""
import ast,argparse,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'performance'),str(ROOT/'performance/tools')]
from stage_layer_mlp_compile import build
from spatial.device_control import adapt_mixer

def main():
 p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
 if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
 attempt='device-control-sim-'+a.attempt;out=ROOT/'performance/evidence'/attempt
 if out.exists():raise ValueError('Frozen attempt')
 active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
 if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
 original,_,_,profiles=build('layer_00',shared_inputs=True,norm_bridge=True,mixer_projections=True)
 selected=next(p for p in profiles if p['pe']==[103,145]);assert selected['parameters']['bank_words']==8130
 files={n:original[n] for n in ['mixer_projection.csl','mixer_native.csl','bf16_row.csl','fp8_shape.csl','fp8_unpack_shift.csl','qwen_math.csl','source_gate.py','elf_inventory.py','check_sram.py','placement.py']}
 for n in ['gateway.csl','prepare.py','run.py']:files[n]=(ROOT/'performance/probes/device_control'/n).read_bytes()
 files['device_control.csl']=(ROOT/'performance/csl/device_control.csl').read_bytes()
 files['peer.csl'],adaptation=adapt_mixer(original[selected['source']],selected['source'])
 files['device-adaptation.json']=(json.dumps(adaptation,indent=2)+'\n').encode()
 files['backend.py']=(ROOT/'runtime/backend.py').read_bytes()
 index=json.loads((ROOT/'performance/evidence/layer-mlp-hw-008/bank-index.json').read_text())
 record=next(r for r in index['records'] if r['pe']==selected['pe'])
 setups=json.loads(original['mixer-setups.json']);setup=next(s for pe,s in setups if pe==selected['pe'])
 meta=json.loads((ROOT/'performance/evidence/layer-mlp-hw-008/fixture.json').read_text())
 dispatch=json.loads((ROOT/'performance/evidence/layer-mlp-reference-002/dispatch.json').read_text())
 if dispatch['physical'] is not False or not dispatch['remote'].startswith('/srv/model-storage/qwen38-singlewse/runs/'):raise ValueError('Original fixture location')
 files['peer.json']=(json.dumps(dict(selected,bank_record=record,setup=setup,original_banks_sha256=meta['hashes']['banks.npy'],
  original_banks_path=dispatch['remote']+'/banks.npy',full_selected_bank=True,complete_stage=False),indent=2)+'\n').encode()
 params=','.join('.%s=%s'%(k,str(v).lower()) for k,v in selected['parameters'].items())
 files['layout.csl']=('''const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {
 @set_rectangle(1,2);
 @set_tile_code(0,0,"gateway.csl",.{.memcpy_params=memcpy.get_params(0)});
 @set_tile_code(0,1,"peer.csl",.{.memcpy_params=memcpy.get_params(0),PARAMS});
 @set_color_config(0,0,@get_color(12),.{.routes=.{.rx=.{RAMP},.tx=.{SOUTH}}});
 @set_color_config(0,1,@get_color(12),.{.routes=.{.rx=.{NORTH},.tx=.{RAMP}}});
 @set_color_config(0,1,@get_color(13),.{.routes=.{.rx=.{RAMP},.tx=.{NORTH}}});
 @set_color_config(0,0,@get_color(13),.{.routes=.{.rx=.{SOUTH},.tx=.{RAMP}}});
 @export_name("command",[*]u32,false);@export_name("response",[*]u32,false);@export_name("audit",[*]u32,false);
 @export_name("exchange",fn(u16,u16)void);
}
'''.replace('PARAMS',params)).encode()
 # Reuse only the already exercised bounded lifecycle; no old test semantics.
 execute=(ROOT/'performance/evidence/mixer-ingress-sim-001/source/execute.py').read_bytes()
 execute=execute.replace(b'--fabric-dims=8,3',b'--fabric-dims=8,4').replace(b'application=(1,1),fabric=(8,3)',b'application=(1,2),fabric=(8,4)')
 ast.parse(execute);files['execute.py']=execute;files['stager.py']=Path(__file__).read_bytes()
 files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
 payload=io.BytesIO()
 with tarfile.open(fileobj=payload,mode='w') as archive:
  for n,b in files.items():
   item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
 remote='/srv/model-storage/qwen38-singlewse/runs/'+attempt
 subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
 unit='qwen38-single-'+attempt
 cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64',
      '--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=380','--property=TimeoutStopSec=5',
      '--property=KillMode=control-group','--property=LimitFSIZE=67108864','--property=LimitCORE=0','--working-directory='+remote,
      '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
      '/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
 subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
 out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
 (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,memory_max=2<<30,swap_max=0,runtime_max_seconds=380,phase_limit_seconds=120),indent=2)+'\n')
 print(json.dumps(dict(remote=remote,unit=unit,physical=False)))

if __name__=='__main__':main()
