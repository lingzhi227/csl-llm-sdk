"""Bounded actual CSL comparison of original and ingress-fused mixer arithmetic."""
import argparse,ast,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.mixer_projection import lower_mixer_projections

def main():
 p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
 if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
 name='mixer-ingress-sim-'+a.attempt;out=ROOT/'performance/evidence'/name
 if out.exists():raise ValueError('Frozen attempt')
 active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
 if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
 files={n:(ROOT/'performance/probes/mixer_ingress'/n).read_bytes() for n in ['pe.csl','prepare.py','run.py']}
 for n in ['mixer_native.csl','mixer_operand.csl','bf16_row.csl','fp8_unpack_shift.csl','mlp_fused.csl','fp8_encode.csl']:
  files[n]=(ROOT/'performance/csl'/n).read_bytes()
 files['fp8_shape.csl']=(ROOT/'performance/probes/native_shapes/fp8_shape.csl').read_bytes()
 files['qwen_math.csl']=(ROOT/'csl/qwen_math.csl').read_bytes()
 prior=ROOT/'performance/evidence/layer-backend-compile-012/source'
 baseline=(prior/'mixer_native.csl').read_bytes()
 files['baseline_native.csl']=baseline.replace(b'"bf16_row.csl"',b'"baseline_bf16_row.csl"')
 files['baseline_bf16_row.csl']=(prior/'bf16_row.csl').read_bytes()
 files['baseline-source.json']=(json.dumps(dict(prior='layer-backend-compile-012',original_native_sha256=hashlib.sha256(baseline).hexdigest(),
  adaptation='Rename only the imported original BF16 module to preserve its exact original source independently of the new strided module.',
  full_original_program=False,physical=False),indent=2)+'\n').encode()
 for n in ['mixer_operand.py','mlp_oracle.py']:files['reference/'+n]=(ROOT/'performance/reference'/n).read_bytes()
 for n in ['backend.py','source_gate.py','elf_inventory.py','weights.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
 for n in ['check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
 files['tensors.json']=(ROOT/'configs/tensors.json').read_bytes()
 plan=json.loads((ROOT/'performance/evidence/layer-native-schedule-002/layer-schedule.json').read_text())
 stage=next(s for s in plan['stages'] if s['id']=='layer_00');region=next(r for r in stage['regions'] if r['role']=='mix')
 lowered=lower_mixer_projections(region,stage['request_controller']['pe'])
 files['region.json']=(json.dumps(dict(matrices=region['matrices'],workers=[lowered['workers'][r] for r in [320,641,39]]),indent=2)+'\n').encode()
 files['layout.csl']=b'''const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout {
 @set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});
 @export_name("bank",[*]u32,false);@export_name("input",[*]u32,false);@export_name("packet",[*]u32,false);
 @export_name("setup",[*]u16,false);@export_name("output",[*]f32,false);@export_name("audit",[*]u32,false);
 @export_name("measure",fn(u32,u16,u16)void);
}
'''
 files['execute.py']=b'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[('prepare',['/usr/bin/python3','prepare.py']),('compile',[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out']),('run',[sdk+'cs_python','run.py'])]
try:
 for phase,cmd in commands:
  with Path(phase+'.log').open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+120
   try:
    while proc.poll() is None:
     if time.monotonic()>deadline:raise TimeoutError(phase+' deadline')
     if phase=='run' and Path('sim.log').exists():
      if Path('sim.log').stat().st_size>8388608:raise RuntimeError('Simulator log bound')
      if 'FATAL:' in Path('sim.log').read_text():raise RuntimeError('Simulator fatal')
     time.sleep(.2)
    if proc.returncode:raise RuntimeError(phase+' exit '+str(proc.returncode))
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
  if phase=='compile':check(Path.cwd(),application=(1,1),fabric=(8,3))
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\\n');raise
'''
 ast.parse(files['execute.py']);files['stager.py']=Path(__file__).read_bytes()
 files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
 payload=io.BytesIO()
 with tarfile.open(fileobj=payload,mode='w') as archive:
  for n,b in files.items():
   item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
 remote='/srv/model-storage/qwen38-singlewse/runs/'+name
 subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
 unit='qwen38-single-'+name
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
