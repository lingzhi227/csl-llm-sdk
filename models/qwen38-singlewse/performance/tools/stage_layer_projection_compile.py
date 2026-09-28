"""Compile actual native-network roles under bounded workstation resources.

Profiles come from the original layer0 gate/up and down resident owner groups.
This tests code/SDK/bank/stack composition only, not the still-open external
ingress, root consumer routes, network liveness or numerical execution.
"""
import argparse,ast,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_routes import contraction_groups,worker


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
    if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
    name='layer-projection-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    path=ROOT/'performance/evidence/layer-native-schedule-002/layer-schedule.json'
    raw_plan=path.read_bytes();selected=json.loads(raw_plan)
    profiles={};covered=0
    for region in [r for s in selected['stages'] for r in s['regions']]:
        if region['role'] not in ('gate_up','down'):continue
        groups=list(contraction_groups(region))
        for group in groups:
            for node in group['nodes']:
                w=worker(region,group,node);p=w['parameters']
                key=(region['role'],p['max_parts'],p['root'],p['children'])
                covered+=1
                # Preserve every weight/auxiliary byte actually assigned to the
                # PE. Select maximum resident payload in each compile-body class.
                score=lambda p:(p['bank_words'],p['subtree_k'])
                if key not in profiles or score(p)>score(profiles[key]['parameters']):profiles[key]=w
    profiles=list(profiles.values());width=len(profiles)
    files={}
    files['native-schedule-binding.json']=(json.dumps(dict(source='performance/evidence/layer-native-schedule-002/layer-schedule.json',sha256=hashlib.sha256(raw_plan).hexdigest(),workers_censused=covered,selection='Maximum actual original matrix plus live auxiliary payload in every selected body class; no bank bytes omitted.'),indent=2)+'\n').encode()
    for dest,source in {
        'layer_projection.csl':'performance/runtime/layer_projection.csl','layer_native.csl':'performance/csl/layer_native.csl',
        'fp8_unpack_shift.csl':'performance/csl/fp8_unpack_shift.csl','fp8_shape.csl':'performance/probes/native_shapes/fp8_shape.csl',
        'qwen_math.csl':'csl/qwen_math.csl','source_gate.py':'runtime/source_gate.py','elf_inventory.py':'runtime/elf_inventory.py',
        'check_sram.py':'performance/runtime/check_sram.py','placement.py':'performance/runtime/placement.py',
        'layer_routes.py':'performance/spatial/layer_routes.py','layer_schedule.py':'performance/spatial/layer_schedule.py',
        'selected-layer0-1.json':'performance/evidence/layer-native-schedule-002/selected-layer0-1.json',
    }.items():files[dest]=(ROOT/source).read_bytes()
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});','layout {',f' @set_rectangle({width},1);']
    for x,profile in enumerate(profiles):
        params=','.join('.%s=%s'%(k,str(v).lower()) for k,v in profile['parameters'].items())
        layout.append(f' @set_tile_code({x},0,"layer_projection.csl",.{{.memcpy_params=memcpy.get_params({x}),{params}}});')
    layout+=[' @export_name("bank",[*]u32,false);',' @export_name("audit",[*]u32,false);',' @export_name("setup",[*]u16,false);',
             ' @export_name("arm",fn(u32)void);','}']
    files['layout.csl']=('\n'.join(layout)+'\n').encode()
    files['profiles.json']=(json.dumps(dict(profiles=profiles,application=[width,1],physical=False,
        scope='Actual selected resident native-network roles; no connected fabric, simulator or neural execution.'),indent=2)+'\n').encode()
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
    script=script.replace('APPLICATION_WIDTH',str(width)).replace('FABRIC_WIDTH',str(width+7));ast.parse(script);files['execute.py']=script.encode()
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
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
    print(json.dumps(dict(remote=remote,unit=unit,profiles=len(profiles),physical=False)))


if __name__=='__main__':main()
