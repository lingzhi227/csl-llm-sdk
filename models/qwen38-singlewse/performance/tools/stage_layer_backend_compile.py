"""Bounded selected-role compilation for the resident layer0/1 backend.

No simulator/model execution and no WSE allocation. This gate composes native
row loops, actual resident payload, request leases and GDN page arithmetic;
network, preprocessing and full-layer composition remain separate open gates.
"""
import argparse,ast,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
    if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
    name='layer-backend-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    files={}
    for dest,source in {
        'layer_fusion_ingress.csl':'performance/csl/layer_fusion_ingress.csl',
        'layer_controller.csl':'performance/runtime/layer_controller.csl','layer_backend.csl':'performance/runtime/layer_backend.csl','layer_native.csl':'performance/csl/layer_native.csl',
        'bf16_row.csl':'performance/csl/bf16_row.csl','layer_fusion.csl':'performance/csl/layer_fusion.csl','mlp_fused.csl':'performance/csl/mlp_fused.csl','fp8_encode.csl':'performance/csl/fp8_encode.csl','qwen_math.csl':'csl/qwen_math.csl','pipeline_lease.csl':'performance/csl/pipeline_lease.csl','gdn_page.csl':'performance/csl/gdn_page.csl',
        'fp8_unpack_shift.csl':'performance/csl/fp8_unpack_shift.csl','fp8_shape.csl':'performance/probes/native_shapes/fp8_shape.csl',
        'source_gate.py':'runtime/source_gate.py','elf_inventory.py':'runtime/elf_inventory.py',
        'check_sram.py':'performance/runtime/check_sram.py','placement.py':'performance/runtime/placement.py',
    }.items():files[dest]=(ROOT/source).read_bytes()
    profiles=[dict(rows=2,columns=128,branches=1,mix=True),dict(rows=8,columns=32,branches=2,mix=False),
              dict(rows=4,columns=64,branches=1,mix=False),dict(rows=2,columns=128,branches=1,mix=False)]
    for profile in profiles:profile.update(bank_words=8814,fusion_actor=False)
    profiles.append(dict(rows=8,columns=32,branches=2,mix=False,bank_words=4160,fusion_actor=True))
    worker_width=len(profiles);width=worker_width+1
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});','layout {',f' @set_rectangle({width},1);']
    for x,profile in enumerate(profiles):
        params=','.join('.%s=%s'%(k,str(v).lower()) for k,v in profile.items())
        layout.append(f' @set_tile_code({x},0,"layer_backend.csl",.{{.memcpy_params=memcpy.get_params({x}),{params}}});')
    layout.append(f' @set_tile_code({worker_width},0,"layer_controller.csl",.{{.memcpy_params=memcpy.get_params({worker_width}),.requests=2,.context=96,.chunks=40}});')
    for symbol,kind in [('bank','u32'),('packet','u32'),('info','u32'),('state_operands','f32'),('state_result','f32'),('partial','f32'),('control','u32'),('fusion_packet','u32'),('bf16_result','f32'),('projection_credit','u32')]:
        layout.append(f' @export_name("{symbol}",[*]{kind},false);')
    for symbol,args in [('command','u16'),('request_begin','u16,u32,u16'),('request_chunk','u16,u32,u16,u32,u16,u16'),
                        ('request_committed','u16,u32'),('request_completed','u16,u32,u16,u16'),('request_reset','u16,u32,u16')]:
        layout.append(f' @export_name("{symbol}",fn({args})void);')
    files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
    files['profiles.json']=(json.dumps(dict(profiles=profiles,application=[width,1],controller={'source':'layer_controller.csl','weight_payload_bytes':0},payload_bytes=35256,
          scope='Selected backend resource census only; no fabric or full-layer execution',physical=False),indent=2)+'\n').encode()
    script=r'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=FABRIC_WIDTH,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=180)
   if code:raise RuntimeError('Compiler exit '+str(code))
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 result=check(Path.cwd(),application=(APPLICATION_WIDTH,1),fabric=(FABRIC_WIDTH,3));verify()
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,executed=False,application=[APPLICATION_WIDTH,1],sram_passed=result['passed'],seconds=time.monotonic()-started))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
'''
    script=script.replace('APPLICATION_WIDTH',str(width)).replace('FABRIC_WIDTH',str(width+7))
    ast.parse(script);files['execute.py']=script.encode()
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
    unit='qwen38-single-'+name
    cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64',
         '--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=210','--property=TimeoutStopSec=5',
         '--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
    print(json.dumps(dict(remote=remote,unit=unit,physical=False)))


if __name__=='__main__':main()
