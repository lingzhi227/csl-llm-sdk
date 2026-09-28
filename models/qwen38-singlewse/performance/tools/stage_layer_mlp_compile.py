"""Compile one fully connected original MLP component with all resident banks.

This is compiler/resource admission, not numerical or model execution. The
mixer/state banks remain resident but inactive. No WSE allocation is requested.
"""
import argparse,ast,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_mlp_network import build_mlp_network,worker_profiles,emit_routes
from spatial.layer_schedule import rank_xy
from spatial.layer_fused_routes import translate_routes


def build(stage_name):
    evidence=ROOT/'performance/evidence';schedule=evidence/'layer-native-schedule-002/layer-schedule.json'
    plan=json.loads(schedule.read_text());stage=next(s for s in plan['stages'] if s['id']==stage_name)
    gate=next(r for r in stage['regions'] if r['role']=='gate_up');shape='x'.join(map(str,gate['rect'][2:]))
    route_file=evidence/f'layer-fused-routes-003/routes-{shape}.json';template=json.loads(route_file.read_text())
    original=next(r for s in plan['stages'] for r in s['regions'] if r['id']==template['region'])
    routes=translate_routes(template,original,gate);network=build_mlp_network(stage,routes)
    profiles=worker_profiles(stage,routes,network);ox,oy,width,height=stage['rect']
    mixer=next(r for r in stage['regions'] if r['role']=='mix')
    for c in mixer['banks']['classes']:
        for rank in range(c['rank_start'],c['rank_end']):
            pe=rank_xy(mixer,rank)
            prefix=c['page_prefix']+(rank-c['rank_start'])*c['auxiliary_pages']
            live=max(0,min(c['auxiliary_pages'],mixer['auxiliary_pages']-prefix))
            payload=c['fp8_slots']*260+c['bf16_slots']*256+live*128
            if pe==stage['request_controller']['pe']:
                if payload:raise ValueError('Controller displaces an original bank')
                continue
            profiles.append(dict(pe=pe,parameters=dict(bank_words=payload//4),source='layer_mlp_standby.csl'))
    profiles.append(dict(pe=stage['request_controller']['pe'],parameters=dict(gate_tail=network['gate_tail_workers'],down_tail=network['down_tail_workers']),source='layer_mlp_controller.csl'))
    if len(profiles)!=width*height or len({tuple(p['pe']) for p in profiles})!=width*height:raise ValueError('Whole stage coverage')
    files={}
    module_files=['layer_native','layer_fusion_transport','layer_fusion_ingress','layer_fusion','layer_mlp_sender','mlp_fused','fp8_encode','fp8_unpack_shift']
    for name in module_files:files[name+'.csl']=(ROOT/f'performance/csl/{name}.csl').read_bytes()
    for name in ['layer_projection','layer_mlp_controller','layer_mlp_standby']:files[name+'.csl']=(ROOT/f'performance/runtime/{name}.csl').read_bytes()
    for dest,source in {'fp8_shape.csl':'performance/probes/native_shapes/fp8_shape.csl','qwen_math.csl':'csl/qwen_math.csl',
                        'source_gate.py':'runtime/source_gate.py','elf_inventory.py':'runtime/elf_inventory.py',
                        'check_sram.py':'performance/runtime/check_sram.py','placement.py':'performance/runtime/placement.py',
                        'layer_mlp_network.py':'performance/spatial/layer_mlp_network.py','layer_fused_routes.py':'performance/spatial/layer_fused_routes.py',
                        'layer_routes.py':'performance/spatial/layer_routes.py','layer_schedule.py':'performance/spatial/layer_schedule.py'}.items():files[dest]=(ROOT/source).read_bytes()
    grants=network['grant_schedule'];fields={'targets':'target','kinds':'kind','indices':'index','words':'words','firsts':'first_row','rows':'rows'}
    files['mlp_schedule.csl']=('\n'.join('const %s=[186]u16{%s};'%(name,','.join(str(g.get(key,0)) for g in grants)) for name,key in fields.items())+'\n').encode()
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}}});','layout {',f' @set_rectangle({width},{height});']
    classes={};setups=[]
    for w in profiles:
        key=(w['source'],json.dumps(w['parameters'],sort_keys=True))
        if key not in classes:classes[key]=dict(source=w['source'],parameters=w['parameters'],pes=[])
        classes[key]['pes'].append(w['pe'])
        if 'arm' in w:setups.append([w['pe'],list(w['arm'].values())])
    for i,c in enumerate(classes.values()):
        coords=[v for x,y in c['pes'] for v in [x-ox,y-oy]];name=f'program_{i}'
        layout.append(' const %s=[%d]u16{%s};'%(name,len(coords),','.join(map(str,coords))))
        params=','.join('.%s=%s'%(k,(v if isinstance(v,str) else str(v).lower())) for k,v in c['parameters'].items())
        layout.append(' for(@range(u16,%d))|i|{@set_tile_code(%s[2*i],%s[2*i+1],"%s",.{.memcpy_params=memcpy.get_params(%s[2*i]),%s});}'%(len(coords)//2,name,name,c['source'],name,params))
    layout.append(emit_routes(network))
    layout += [' @export_name("%s",[*]%s,false);'%(n,t) for n,t in [('bank','u32'),('audit','u32'),('setup','u16'),('fusion_signal','u32'),('quant_input','u32'),('mlp_output','u32'),('sender_status','u32'),('silu_lut','u16'),('native_input','u16'),('native_scales','f32'),('ticks','u16')]]
    layout += [' @export_name("arm",fn(u32)void);',' @export_name("start",fn()void);',' @export_name("finish",fn()void);',' @export_name("fusion_step",fn(u16,u32,u16,u16)void);','}']
    files['layout.csl']=('\n'.join(layout)+'\n').encode()
    files['profiles.json']=(json.dumps(dict(profiles=list(classes.values()),application=[width,height],physical=False,executed=False,
        whole_stage=stage_name,scope='Original connected MLP component; mixer/state banks retained but inactive; no full-layer or model execution.'),separators=(',',':'))+'\n').encode()
    files['worker-setups.json']=(json.dumps(setups,separators=(',',':'))+'\n').encode()
    files['network-binding.json']=(json.dumps(dict(stage=stage_name,audit=network['audit'],senders=network['senders'],sinks=network['down_sinks'],grant_schedule=grants,
        network_sha256=hashlib.sha256(json.dumps(network,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        source_bindings={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [schedule,route_file]}),indent=2)+'\n').encode()
    return files,width,height,profiles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--stage',default='layer_00');args=parser.parse_args()
    if len(args.attempt)!=3 or not args.attempt.isdigit():raise ValueError('Attempt')
    name='layer-mlp-compile-'+args.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    files,width,height,profiles=build(args.stage)
    script=r'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=FABRIC_WIDTH,FABRIC_HEIGHT','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=300)
   if code:raise RuntimeError('Compiler exit '+str(code))
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 result=check(Path.cwd(),application=(APPLICATION_WIDTH,APPLICATION_HEIGHT),fabric=(FABRIC_WIDTH,FABRIC_HEIGHT));verify()
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,executed=False,application=[APPLICATION_WIDTH,APPLICATION_HEIGHT],sram_passed=result['passed'],seconds=time.monotonic()-started))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
'''
    script=script.replace('APPLICATION_WIDTH',str(width)).replace('FABRIC_WIDTH',str(width+7)).replace('APPLICATION_HEIGHT',str(height)).replace('FABRIC_HEIGHT',str(height+2));ast.parse(script);files['execute.py']=script.encode()
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
    unit='qwen38-single-'+name
    cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64',
         '--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=330','--property=TimeoutStopSec=5',
         '--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
    print(json.dumps(dict(remote=remote,unit=unit,profiles=len(profiles),physical=False)))


if __name__=='__main__':main()
