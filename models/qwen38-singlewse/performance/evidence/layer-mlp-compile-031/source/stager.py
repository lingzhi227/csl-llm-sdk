"""Compile one fully connected original MLP component with all resident banks.

This is compiler/resource admission, not numerical or model execution. The
mixer/state banks remain resident but inactive. No WSE allocation is requested.
"""
import argparse,ast,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_mlp_network import build_mlp_network,worker_profiles,emit_routes
from spatial.layer_schedule import rank_xy
from spatial.layer_norm_banks import lower_norm_banks
from spatial.layer_fused_routes import translate_routes


DEFAULT_COMPILE_MEMORY_GIB = 16
DEFAULT_COMPILE_SECONDS = 1800
HOST_HEADROOM_GIB = 8


def check_host_memory(available_bytes, memory_gib):
    """Reserve host headroom without silently shrinking a full-layer job."""
    required = (memory_gib + HOST_HEADROOM_GIB) * (1 << 30)
    if available_bytes < required:
        raise ValueError(
            f'Full-layer compile needs {memory_gib} GiB plus '
            f'{HOST_HEADROOM_GIB} GiB host headroom; '
            f'only {available_bytes / (1 << 30):.2f} GiB available. '
            'Reconcile other workloads before dispatch; do not retry with an '
            'undersized cap merely to pass admission.'
        )
    return dict(requested_memory_gib=memory_gib,
                host_available_bytes=available_bytes,
                host_headroom_gib=HOST_HEADROOM_GIB)


def build(stage_name,native_loop="map",shared_inputs=False,norm_bridge=False,mixer_projections=False,joint_banks=False,routed_control=False,dialogue=False):
    if native_loop not in ["map","unroll"]:raise ValueError("Native loop candidate")
    if routed_control and not joint_banks:raise ValueError('Routed control requires the original joint-bank candidate')
    if dialogue and not (joint_banks and routed_control):raise ValueError('Single-dialogue specialization requires the complete routed joint-bank source')
    evidence=ROOT/'performance/evidence';schedule=evidence/'layer-native-schedule-002/layer-schedule.json'
    plan=json.loads(schedule.read_text());stage=next(s for s in plan['stages'] if s['id']==stage_name)
    gate=next(r for r in stage['regions'] if r['role']=='gate_up');shape='x'.join(map(str,gate['rect'][2:]))
    route_file=evidence/f'layer-fused-routes-003/routes-{shape}.json';template=json.loads(route_file.read_text())
    original=next(r for s in plan['stages'] for r in s['regions'] if r['id']==template['region'])
    routes=translate_routes(template,original,gate);network=build_mlp_network(stage,routes,shared_inputs,norm_bridge)
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
            norm=next((s for s in network['senders'] if s.get('norm_bridge') and s['pe']==pe),None)
            if norm:
                bus=next(r for r in network['routes'] if r['color']==18 and r['pe']==pe)
                profiles.append(dict(pe=pe,parameters=dict(bank_words=payload//4,rank=norm['input_group'],sender_id=norm['id'],
                    bus_rx=bus['rx'][0],bus_tx=bus['tx'][0]),source='layer_norm_sender.csl'))
            elif norm_bridge and any(s.get('norm_pe')==pe for s in network['senders']):
                owner=next(s for s in network['senders'] if s.get('norm_pe')==pe)
                profiles.append(dict(pe=pe,parameters=dict(bank_words=payload//4,rank=owner['input_group']),source='layer_norm_bridge.csl'))
            else:profiles.append(dict(pe=pe,parameters=dict(bank_words=payload//4),source='layer_mlp_standby.csl'))
    profiles.append(dict(pe=stage['request_controller']['pe'],parameters=dict(gate_tail=network['gate_tail_workers'],down_tail=network['down_tail_workers'],shared_inputs=shared_inputs,norm_bridge=norm_bridge),source='layer_mlp_controller.csl'))
    if len(profiles)!=width*height or len({tuple(p['pe']) for p in profiles})!=width*height:raise ValueError('Whole stage coverage')
    page_remaps=lower_norm_banks(stage,profiles,norm_bridge)
    files={'state-page-remap.json':(json.dumps(dict(pages=page_remaps,scope='Only these original recurrent state pages relocate; matrix ownership and total state capacity unchanged.'),indent=2)+'\n').encode()}
    module_files=['layer_native','layer_fusion_transport','layer_fusion_ingress','layer_fusion','layer_mlp_sender','mlp_fused','fp8_encode','fp8_unpack_shift']
    for name in module_files:files[name+'.csl']=(ROOT/f'performance/csl/{name}.csl').read_bytes()
    for name in ['layer_projection','layer_mlp_controller','layer_mlp_standby','layer_norm_bridge','layer_norm_sender']:files[name+'.csl']=(ROOT/f'performance/runtime/{name}.csl').read_bytes()
    for dest,source in {'fp8_shape.csl':'performance/probes/native_shapes/fp8_shape.csl','qwen_math.csl':'csl/qwen_math.csl',
                        'source_gate.py':'runtime/source_gate.py','elf_inventory.py':'runtime/elf_inventory.py',
                        'check_sram.py':'performance/runtime/check_sram.py','placement.py':'performance/runtime/placement.py',
                        'layer_mlp_network.py':'performance/spatial/layer_mlp_network.py','layer_fused_routes.py':'performance/spatial/layer_fused_routes.py',
                        'layer_norm_banks.py':'performance/spatial/layer_norm_banks.py','layer_routes.py':'performance/spatial/layer_routes.py','layer_schedule.py':'performance/spatial/layer_schedule.py'}.items():files[dest]=(ROOT/source).read_bytes()
    if native_loop=='unroll':files['fp8_shape.csl']=(ROOT/'performance/probes/native_shapes/fp8_unroll.csl').read_bytes()
    grants=network['grant_schedule'];fields={'targets':'target','kinds':'kind','indices':'index','words':'words','firsts':'first_row','rows':'rows'}
    preparations=network['prepare_schedule']
    schedules=['const %s=[%d]u16{%s};'%(name,len(grants),','.join(str(g.get(key,0)) for g in grants)) for name,key in fields.items()]
    schedules+=['const prepare_%s=[%d]u16{%s};'%(name,len(preparations),','.join(str(g[key]) for g in preparations)) for name,key in {'targets':'target','indices':'index'}.items()]
    files['mlp_schedule.csl']=('\n'.join(schedules)+'\n').encode()
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
    layout += [' @export_name("%s",[*]%s,false);'%(n,t) for n,t in [('bank','u32'),('audit','u32'),('setup','u16'),('fusion_signal','u32'),('quant_input','u32'),('mlp_output','u32'),('sender_status','u32'),('silu_lut','u16'),('native_input','u16'),('native_scales','f32'),('ticks','u16'),('transport_stats','u32')]]
    if norm_bridge:layout += [' @export_name("%s",[*]u32,false);'%n for n in ['residual_input','norm_gains','successor_output','bridge_stats','norm_pre_output']]
    layout += [' @export_name("arm",fn(u32)void);',' @export_name("start",fn()void);',' @export_name("finish",fn()void);',' @export_name("fusion_step",fn(u16,u32,u16,u16)void);','}']
    files['layout.csl']=('\n'.join(layout)+'\n').encode()
    files['profiles.json']=(json.dumps(dict(profiles=list(classes.values()),application=[width,height],physical=False,executed=False,
        whole_stage=stage_name,scope=('Residual/RMS -> complete original MLP -> residual/successor RMS; mixer/state banks retained but neural mixer inactive; numerical execution unqualified.' if norm_bridge else 'Original connected MLP component; mixer/state banks retained but inactive; no full-layer or model execution.')),separators=(',',':'))+'\n').encode()
    files['worker-setups.json']=(json.dumps(setups,separators=(',',':'))+'\n').encode()
    files['network-binding.json']=(json.dumps(dict(stage=stage_name,audit=network['audit'],senders=network['senders'],sinks=network['down_sinks'],grant_schedule=grants,prepare_schedule=preparations,
        distribution_packets=network['distribution_packets'],shared_inputs=shared_inputs,norm_bridge=norm_bridge,controller_transport=network['controller_transport'],
        network_sha256=hashlib.sha256(json.dumps(network,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        source_bindings={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [schedule,route_file]}),indent=2)+'\n').encode()
    if mixer_projections:
        if not norm_bridge:raise ValueError('Mixer composition requires the accepted norm+MLP base')
        if joint_banks:
            from spatial.joint_placement import lower_joint_banks
            calibration=evidence/'joint-placement-calibration-001.json'
            cohost=evidence/'device-control-calibration-001.json'
            region,placement=lower_joint_banks(stage,profiles,json.loads(calibration.read_text()),json.loads(cohost.read_text()),mix_extra_reserve=256 if routed_control else 0)
            stage['regions']=[region if r['role']=='mix' else r for r in stage['regions']]
            files['joint-bank-placement.json']=(json.dumps(placement,separators=(',',':'))+'\n').encode()
            files['joint-stage.json']=(json.dumps(stage,separators=(',',':'))+'\n').encode()
            files['joint_placement.py']=(ROOT/'performance/spatial/joint_placement.py').read_bytes()
            files['joint-calibration.json']=calibration.read_bytes()
            files['joint-cohost-calibration.json']=cohost.read_bytes()
            if dialogue:
                from spatial.dialogue_placement import lower_dialogue_banks
                single=lower_dialogue_banks(mixer,region,placement,profiles)
                files['dialogue-bank-placement.json']=(json.dumps(single,separators=(',',':'))+'\n').encode()
                files['dialogue_placement.py']=(ROOT/'performance/spatial/dialogue_placement.py').read_bytes()
                files['bank-specialization.json']=(json.dumps(dict(active='dialogue-bank-placement.json',
                    source='joint-bank-placement.json',logical_metadata='joint-stage.json',
                    original_page_ids_preserved=True,removed_request=1,retained_request=0,work_slots=2,
                    scope='Bank specialization only; source logical metadata still describes both old contexts. Use DialogueAuxiliaryPlacement for every live auxiliary address. No dialogue runtime or speed admission.'),indent=2)+'\n').encode()
        from spatial.mixer_composition import compose_mixer
        result=compose_mixer(files,width,height,profiles,stage,network,device_control=joint_banks,routed_control=routed_control)
        if dialogue:
            metadata=json.loads(files['profiles.json'])
            metadata.update(bank_specialization='dialogue-bank-placement.json',conversations=1,work_slots=2)
            files['profiles.json']=(json.dumps(metadata,separators=(',',':'))+'\n').encode()
        return result
    if joint_banks:raise ValueError('Joint bank placement requires complete mixer composition')
    return files,width,height,profiles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--stage',default='layer_00');parser.add_argument('--native-loop',choices=['map','unroll'],default='map');parser.add_argument('--shared-inputs',action='store_true');parser.add_argument('--norm-bridge',action='store_true');parser.add_argument('--mixer-projections',action='store_true');parser.add_argument('--joint-banks',action='store_true');parser.add_argument('--routed-control',action='store_true');parser.add_argument('--dialogue',action='store_true');parser.add_argument('--frontend',action='store_true');parser.add_argument('--compact-mlp',action='store_true');parser.add_argument('--gdn-columns',action='store_true');parser.add_argument('--gdn-refine',action='store_true');parser.add_argument('--resident-gdn',action='store_true');parser.add_argument('--resident-gdn-bridge',action='store_true');parser.add_argument('--resident-gdn-refine',action='store_true');parser.add_argument('--resident-gdn-groups',type=int,nargs='*',default=[]);parser.add_argument('--compile-seconds',type=int,choices=[300,600,900,1800,3600],default=DEFAULT_COMPILE_SECONDS);parser.add_argument('--memory-gib',type=int,choices=[2,3,4,6,8,12,16,20,24],default=DEFAULT_COMPILE_MEMORY_GIB);args=parser.parse_args()
    if len(args.attempt)!=3 or not args.attempt.isdigit():raise ValueError('Attempt')
    name='layer-mlp-compile-'+args.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    if args.gdn_refine and not args.gdn_columns:raise ValueError('GDN refinement requires full GDN builder')
    if (args.resident_gdn_groups or args.resident_gdn_refine) and not args.resident_gdn:
        raise ValueError('Production groups require resident GDN builder')
    if args.resident_gdn_bridge:
        if args.resident_gdn or args.gdn_columns or args.frontend or args.compact_mlp or args.resident_gdn_groups or args.resident_gdn_refine:
            raise ValueError('Choose the frozen bridge builder alone')
        from tools.build_resident_bridge_stage import build as build_resident_bridge
        files,width,height,profiles=build_resident_bridge()
        args.routed_control=True
    elif args.resident_gdn:
        if args.gdn_columns or args.frontend or args.compact_mlp:
            raise ValueError('Choose one frozen builder')
        from tools.build_resident_gdn_stage import build as build_resident_gdn
        files,width,height,profiles=build_resident_gdn(routed_groups=tuple(args.resident_gdn_groups),refine_banks=args.resident_gdn_refine)
        args.routed_control=True
    elif args.gdn_columns:
        if args.frontend or args.compact_mlp:raise ValueError('Choose one frozen builder')
        from tools.build_gdn_stage import build as build_gdn
        files,width,height,profiles=build_gdn(refine=args.gdn_refine)
        args.routed_control=True
    elif args.compact_mlp:
        if args.frontend:raise ValueError("Choose a single frozen builder")
        from tools.build_compact_mlp_stage import build as build_compact_mlp
        files,width,height,profiles=build_compact_mlp()
        args.routed_control=True
    elif args.frontend:
        from tools.build_frontend_stage import build as build_frontend
        files,width,height,profiles=build_frontend()
        args.routed_control=True
    else:
        files,width,height,profiles=build(args.stage,args.native_loop,args.shared_inputs,args.norm_bridge,args.mixer_projections,args.joint_banks,args.routed_control,args.dialogue)
    script=r'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=FABRIC_WIDTH,FABRIC_HEIGHT','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=COMPILE_SECONDS)
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
    compile_seconds=args.compile_seconds
    script=script.replace('APPLICATION_WIDTH',str(width)).replace('FABRIC_WIDTH',str(width+7)).replace('APPLICATION_HEIGHT',str(height)).replace('FABRIC_HEIGHT',str(height+2)).replace('COMPILE_SECONDS',str(compile_seconds))
    if args.routed_control:script=script.replace("'--arch=wse3'","'--arch=wse3','--params=MEMCPYH2D_DATA_1_ID:14,MEMCPYD2H_DATA_1_ID:15'")
    ast.parse(script);files['execute.py']=script.encode()
    files['stager.py']=Path(__file__).read_bytes()
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
    unit='qwen38-single-'+name
    probe = "from pathlib import Path; print(next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))"
    available_bytes = int(subprocess.check_output(
        ['ssh','workstation','python3 -c '+shlex.quote(probe)],text=True,timeout=20).strip())
    resource_admission = check_host_memory(available_bytes, args.memory_gib)
    cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax='+str(args.memory_gib)+'G','--property=MemorySwapMax=0','--property=TasksMax=64',
         '--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec='+str(compile_seconds+30),'--property=TimeoutStopSec=5',
         '--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,
        memory_max=args.memory_gib*(1<<30),swap_max=0,compiler_seconds=compile_seconds,
        service_seconds=compile_seconds+30,resource_admission=resource_admission),indent=2)+'\n')
    print(json.dumps(dict(remote=remote,unit=unit,profiles=len(profiles),physical=False)))


if __name__=='__main__':main()
