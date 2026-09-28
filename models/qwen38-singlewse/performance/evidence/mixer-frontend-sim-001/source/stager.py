"""Bounded original-parameter native-arrival qualification of all16 frontends."""
import argparse,ast,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    if len(args.attempt)!=3 or not args.attempt.isdigit():raise ValueError('Attempt')
    name='mixer-frontend-sim-'+args.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    live=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if live.stdout.strip():raise ValueError('Live workstation owner: '+live.stdout)
    source=ROOT/'performance/evidence/layer-mlp-compile-022/source';manifest=json.loads((source.parent/'source-manifest.json').read_text())['files']
    files={n:(source/n).read_bytes() for n in ['mixer_frontend_native.csl','qwen_math.csl','frontend-network.json','mixer-setups.json']}
    for n,b in files.items():
        if hashlib.sha256(b).hexdigest()!=manifest[n]:raise ValueError('Admitted source identity changed')
    port=(ROOT/'performance/runtime/mixer_frontend_native_port.cslpart').read_bytes()
    if (source/'frontend_native_001_device_mixer_layer_mlp_standby.csl').read_bytes().count(port)!=1:
        raise ValueError('Native arrival port differs from admitted full-stage source')
    for n in ['consumer.csl','source.csl','prepare.py','run.py']:files[n]=(ROOT/'performance/probes/mixer_frontend'/n).read_bytes()
    files['consumer.csl']=files['consumer.csl'].replace(b'FRONTEND_NATIVE_PORT',port)
    files['reference/__init__.py']=b'';files['reference/frontend_oracle.py']=(ROOT/'performance/reference/frontend_oracle.py').read_bytes()
    original_helper=(ROOT/'experiments/gdn_head/prepare.py').read_bytes()
    files['reference-provenance.json']=(json.dumps(dict(original_interval_reference='experiments/gdn_head/prepare.py',sha256=hashlib.sha256(original_helper).hexdigest(),
        adaptation='Reuse scalar/BF16 interval helpers; new shared Q/K and3-V frontend reference with original complete projection inputs. Frozen recurrent results are injected for gated normalization only.',
        native_port_sha256=hashlib.sha256(port).hexdigest(),compiled_source_manifest_sha256=hashlib.sha256((source.parent/'source-manifest.json').read_bytes()).hexdigest()),indent=2)+'\n').encode()
    for n in ['backend.py','source_gate.py','elf_inventory.py','weights.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
    for n in ['check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
    files['tensors.json']=(ROOT/'configs/tensors.json').read_bytes()
    oracle=ROOT/'performance/evidence/layer-mixer-reference-001';meta=json.loads((oracle/'fixture.json').read_text())
    files['reference-input.json']=(json.dumps(dict(root=json.loads((oracle/'dispatch.json').read_text())['remote'],fixture_sha256=meta['fixture_sha256']),indent=2)+'\n').encode()
    counts=json.loads(files['frontend-network.json'])['consumer_projected_packets']
    layout=['const memcpy=@import_module("<memcpy/get_params>",.{.width=16,.height=2});','layout {',' @set_rectangle(16,2);']
    for group in range(16):
        layout.append(' @set_tile_code(%d,0,"source.csl",.{.memcpy_params=memcpy.get_params(%d)});'%(group,group))
        layout.append(' @set_tile_code(%d,1,"consumer.csl",.{.memcpy_params=memcpy.get_params(%d),.frontend_group=%d,.frontend_projected_packets=%d});'%(group,group,group,counts[group]))
        layout.append(' @set_color_config(%d,0,@get_color(3),.{.routes=.{.rx=.{RAMP},.tx=.{SOUTH}}});'%group)
        layout.append(' @set_color_config(%d,1,@get_color(3),.{.routes=.{.rx=.{NORTH},.tx=.{RAMP}}});'%group)
    for n in ['weights','history','gain','parameters','output']:layout.append(' @export_name("%s",[*]u16,false);'%n)
    for n in ['wire','packet','audit','status']:layout.append(' @export_name("%s",[*]u32,false);'%n)
    for n in ['send','inspect','clear']:layout.append(' @export_name("%s",fn()void);'%n)
    layout.append(' @export_name("begin",fn(u32)void);')
    for n in ['prepare','consume','retire']:layout.append(' @export_name("%s",fn(u16)void);'%n)
    files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
    files['execute.py']=b'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[('prepare',['/usr/bin/python3','prepare.py'],90),('compile',[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=23,4','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],120),('run',[sdk+'cs_python','run.py'],300)]
try:
 for phase,cmd,seconds in commands:
  with Path(phase+'.log').open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+seconds
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
  if phase=='compile':check(Path.cwd(),application=(16,2),fabric=(23,4))
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\\n');raise
'''
    files['stager.py']=Path(__file__).read_bytes()
    for n,b in files.items():
        if n.endswith('.py'):ast.parse(b)
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as tar:
        for n,b in files.items():
            info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
    unit='qwen38-single-'+name
    cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64',
        '--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=540','--property=TimeoutStopSec=5',
        '--property=KillMode=control-group','--property=LimitFSIZE=67108864','--property=LimitCORE=0','--working-directory='+remote,
        '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
        '/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,memory_max=2<<30,swap_max=0,runtime_max_seconds=540),indent=2)+'\n')
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    print(json.dumps(dict(remote=remote,unit=unit,physical=False,application=[16,2])))


if __name__=='__main__':main()
