"""Bounded compile of actual cohost code with declared calibration bank extents.

The full producer/controller is unchanged. Weight-bearing cohosts use4096-word
calibration banks solely to expose code/workspace size after full compilation
failed before producing their ELF images. Never infer full-bank or whole-stage
admission from this sample; every changed payload is recorded explicitly.
"""
import argparse,ast,hashlib,io,json,re,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'performance'),str(ROOT/'performance/tools')]
from stage_layer_mlp_compile import build


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--device-control',action='store_true');a=p.parse_args()
    if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
    name='layer-backend-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    original,_,_,profiles=build('layer_00',shared_inputs=True,norm_bridge=True,mixer_projections=True)
    selected=[next(p for p in profiles if p['source']=='mixer_layer_mlp_controller.csl')]
    for root in (False,True):selected.append(next(p for p in profiles if p['source']=='mixer_layer_mlp_standby.csl' and p['parameters']['mixer_can_root']==root))
    for source in ('mixer_layer_norm_bridge.csl','mixer_layer_norm_sender.csl'):
        for rank in (0,1,39):selected.append(next(p for p in profiles if p['source']==source and p['parameters']['rank']==rank))
    keep=['mixer_layer_mlp_controller.csl','mixer_layer_mlp_standby.csl','mixer_layer_norm_bridge.csl','mixer_layer_norm_sender.csl',
          'mixer_operand.csl','mixer_projection.csl','mixer_native.csl','bf16_row.csl','fp8_shape.csl','fp8_unpack_shift.csl',
          'mlp_fused.csl','fp8_encode.csl','qwen_math.csl','layer_mlp_sender.csl','mlp_schedule.csl',
          'source_gate.py','elf_inventory.py','check_sram.py','placement.py']
    files={n:original[n] for n in keep};width=len(selected);adaptations={}
    if a.device_control:
        from spatial.device_control import adapt_mixer
        files['device_control.csl']=(ROOT/'performance/csl/device_control.csl').read_bytes()
        files['device_control.py']=(ROOT/'performance/spatial/device_control.py').read_bytes()
        for profile in selected[1:]:
            name=profile['source'];adapted='device_'+name
            if adapted not in files:files[adapted],adaptations[adapted]=adapt_mixer(files[name],name)
            profile['source']=adapted
        files['device-adaptation.json']=(json.dumps(adaptations,indent=2)+'\n').encode()
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});','layout {',f' @set_rectangle({width},1);'];samples=[]
    for x,profile in enumerate(selected):
        params=dict(profile['parameters']);actual=params.get('bank_words',0)
        if 'bank_words' in params:params['bank_words']=4096
        fields=','.join('.%s=%s'%(k,str(v).lower() if not isinstance(v,str) else v) for k,v in params.items())
        layout.append(f' @set_tile_code({x},0,"{profile["source"]}",.{{.memcpy_params=memcpy.get_params({x}),{fields}}});')
        samples.append(dict(pe=[x,0],source=profile['source'],parameters=params,original_pe=profile['pe'],original_bank_words=actual,
                            payload_changed=params.get('bank_words',0)!=actual))
    declarations=[s for s in original['layout.csl'].decode().splitlines() if s.strip().startswith('@export_name')]
    if a.device_control:
        exported=set()
        for value in re.findall(r'@export_symbol\(([^;]+)\);',files['mixer_layer_mlp_controller.csl'].decode()):
            exported.add(value.split(',')[-1].strip().strip('"'))
        declarations=[s for s in declarations if re.search(r'@export_name\("([^"]+)"',s).group(1) in exported]
    layout += declarations
    files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
    files['profiles.json']=(json.dumps(dict(samples=samples,application=[width,1],physical=False,executed=False,full_bank_admission=False,device_control=a.device_control,
        scope='Actual producer buffers/code and selected cohost code/storage calibration with4096-word banks. No fabric execution or complete-layer/model admission.'),indent=2)+'\n').encode()
    script=r'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=FABRIC_WIDTH,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=90)
   if code:raise RuntimeError('Compiler exit '+str(code))
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 result=check(Path.cwd(),application=(APPLICATION_WIDTH,1),fabric=(FABRIC_WIDTH,3));verify()
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,executed=False,full_bank_admission=False,full_stage_admission=False,sram_passed=result['passed'],seconds=time.monotonic()-started))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
'''.replace('APPLICATION_WIDTH',str(width)).replace('FABRIC_WIDTH',str(width+7))
    ast.parse(script);files['execute.py']=script.encode();files['stager.py']=Path(__file__).read_bytes()
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
    unit='qwen38-single-'+name
    cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=1G','--property=MemorySwapMax=0','--property=TasksMax=64',
         '--property=CPUQuota=100%','--property=AllowedCPUs=6','--property=RuntimeMaxSec=110','--property=TimeoutStopSec=5',
         '--property=KillMode=control-group','--property=LimitFSIZE=67108864','--property=LimitCORE=0','--working-directory='+remote,
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '/usr/bin/taskset','--cpu-list','6','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
    print(json.dumps(dict(remote=remote,unit=unit,samples=width,physical=False,full_bank_admission=False)))


if __name__=='__main__':main()
