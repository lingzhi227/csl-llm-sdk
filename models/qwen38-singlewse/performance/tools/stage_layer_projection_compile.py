"""Compile actual native-network roles under bounded workstation resources.

Profiles come from the original layer0 gate/up and down resident owner groups.
This tests code/SDK/bank/stack composition only, not the still-open external
ingress, root consumer routes, network liveness or numerical execution.
"""
import argparse,ast,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_routes import contraction_groups,worker
from spatial.layer_fused_routes import compact_tree_groups,root_parameters,actor_parameters,csl_routes,translate_routes


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--fused-routes',type=Path)
    p.add_argument('--whole-region');a=p.parse_args()
    if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
    name='layer-projection-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    path=ROOT/'performance/evidence/layer-native-schedule-002/layer-schedule.json'
    raw_plan=path.read_bytes();selected=json.loads(raw_plan)
    regions=[r for s in selected['stages'] for r in s['regions']]
    if a.whole_region:
        if not a.fused_routes:raise ValueError('A complete checked fabric is required')
        region=next(r for r in regions if r['id']==a.whole_region)
        if region['role']!='gate_up':raise ValueError('Only composed gate/up regions currently supported')
        regions=[region]
    profiles={};covered=0;route_templates={}
    if a.fused_routes:
        audit=json.loads((a.fused_routes/'audit.json').read_text())
        if not audit['passed'] or len(audit['regions'])!=64:raise ValueError('All resident MLP routes required')
        for f in sorted(a.fused_routes.glob('routes-*.json')):
            route=json.loads(f.read_text());route_templates[tuple(route['rect'][2:])]=route
    for region in regions:
        if region['role'] not in ('gate_up','down'):continue
        groups=list(compact_tree_groups(region) if a.fused_routes and region['role']=='gate_up' else contraction_groups(region))
        for group in groups:
            for node in group['nodes']:
                w=worker(region,group,node);p=w['parameters']
                if a.fused_routes and region['role']=='gate_up':p['copy_transport']=route_templates[tuple(region['rect'][2:])].get('copied_transport',False)
                if a.fused_routes and region['role']=='gate_up' and p['root']:
                    p.update(root_parameters(route_templates[tuple(region['rect'][2:])],group))
                key=(region['role'],p['max_parts'],p['root'],p['children'],p.get('boundary_rows',0)>0)
                if a.whole_region:key=tuple(w['pe'])
                covered+=1
                # Preserve every weight/auxiliary byte actually assigned to the
                # PE. Select maximum resident payload in each compile-body class.
                score=lambda p:(p['bank_words'],p['subtree_k'])
                if a.whole_region or key not in profiles or score(p)>score(profiles[key]['parameters']):profiles[key]=w
        if a.fused_routes and region['role']=='gate_up':
            routes=route_templates[tuple(region['rect'][2:])]
            for actor in routes['actors']:
                group=next(g for g in groups if g['rank_start']<=actor['rank']<g['rank_start']+g['workers'])
                node=group['nodes'][actor['rank']-group['rank_start']]
                w=worker(region,group,node);p=w['parameters'];p.update(actor_parameters(routes,groups,actor))
                if p['root']:raise ValueError('Actor conflicts with root queue ownership')
                key=('fusion_actor',p['max_parts'],p['children'],p['actor_has_suffix'])
                if a.whole_region:key=tuple(w['pe'])
                score=lambda p:(p['bank_words'],p['subtree_k'])
                if a.whole_region or key not in profiles or score(p)>score(profiles[key]['parameters']):profiles[key]=w
    profiles=list(profiles.values());width=len(profiles);height=1
    if a.whole_region:
        region=regions[0];width,height=region['rect'][2:]
        if len(profiles)!=width*height:raise ValueError('Complete original region must be assigned once')
    files={}
    if a.fused_routes:
        files['fused-route-binding.json']=(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.fused_routes.glob('routes-*.json'))},indent=2)+'\n').encode()
    files['native-schedule-binding.json']=(json.dumps(dict(source='performance/evidence/layer-native-schedule-002/layer-schedule.json',sha256=hashlib.sha256(raw_plan).hexdigest(),workers_censused=covered,selection='Maximum actual original matrix plus live auxiliary payload in every selected body class; no bank bytes omitted.'),indent=2)+'\n').encode()
    for dest,source in {
        'layer_projection.csl':'performance/runtime/layer_projection.csl','layer_native.csl':'performance/csl/layer_native.csl',
        'layer_fusion_transport.csl':'performance/csl/layer_fusion_transport.csl',
        'layer_fusion_ingress.csl':'performance/csl/layer_fusion_ingress.csl',
        'layer_fusion.csl':'performance/csl/layer_fusion.csl','mlp_fused.csl':'performance/csl/mlp_fused.csl',
        'fp8_encode.csl':'performance/csl/fp8_encode.csl',
        'fp8_unpack_shift.csl':'performance/csl/fp8_unpack_shift.csl','fp8_shape.csl':'performance/probes/native_shapes/fp8_shape.csl',
        'qwen_math.csl':'csl/qwen_math.csl','source_gate.py':'runtime/source_gate.py','elf_inventory.py':'runtime/elf_inventory.py',
        'check_sram.py':'performance/runtime/check_sram.py','placement.py':'performance/runtime/placement.py',
        'layer_routes.py':'performance/spatial/layer_routes.py','layer_schedule.py':'performance/spatial/layer_schedule.py',
        'layer_fused_routes.py':'performance/spatial/layer_fused_routes.py',
        'selected-layer0-1.json':'performance/evidence/layer-native-schedule-002/selected-layer0-1.json',
    }.items():files[dest]=(ROOT/source).read_bytes()
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}}});','layout {',f' @set_rectangle({width},{height});']
    for index,profile in enumerate(profiles):
        x,y=(profile['pe'][0]-region['rect'][0],profile['pe'][1]-region['rect'][1]) if a.whole_region else (index,0)
        params=','.join('.%s=%s'%(k,str(v).lower()) for k,v in profile['parameters'].items())
        layout.append(f' @set_tile_code({x},{y},"layer_projection.csl",.{{.memcpy_params=memcpy.get_params({x}),{params}}});')
    if a.whole_region:
        template=route_templates[tuple(region['rect'][2:])]
        original=next(r for s in selected['stages'] for r in s['regions'] if r['id']==template['region'])
        translated=translate_routes(template,original,region)
        layout.append(csl_routes(translated,region['rect'][:2]))
        files['whole-region-route-audit.json']=(json.dumps(translated['audit'],indent=2)+'\n').encode()
    layout+=[' @export_name("bank",[*]u32,false);',' @export_name("audit",[*]u32,false);',' @export_name("setup",[*]u16,false);',
             ' @export_name("arm",fn(u32)void);',' @export_name("fusion_signal",[*]u32,false);',
             ' @export_name("fusion_step",fn(u16,u32,u16,u16)void);','}']
    files['layout.csl']=('\n'.join(layout)+'\n').encode()
    profile_record=profiles
    if a.whole_region:
        classes={};setups=[]
        for w in profiles:
            key=json.dumps(w['parameters'],sort_keys=True)
            if key not in classes:classes[key]=dict(parameters=w['parameters'],pes=[])
            classes[key]['pes'].append(w['pe']);setups.append([w['pe'],list(w['arm'].values())])
        profile_record=list(classes.values());files['worker-setups.json']=(json.dumps(setups,separators=(',',':'))+'\n').encode()
    files['profiles.json']=(json.dumps(dict(profiles=profile_record,application=[width,height],physical=False,
        whole_region=a.whole_region,
        scope='Complete original gate/up region with simultaneous reduction, projection, copied-credit and fusion-actor transport; input/downstream distributions are open; no execution.' if a.whole_region else
              'Actual selected resident native-network and fusion-actor roles; no connected fabric, simulator or neural execution.'),indent=2)+'\n').encode()
    script=r'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=FABRIC_WIDTH,FABRIC_HEIGHT','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=180)
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
         '--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=210','--property=TimeoutStopSec=5',
         '--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
    print(json.dumps(dict(remote=remote,unit=unit,profiles=len(profiles),physical=False)))


if __name__=='__main__':main()
