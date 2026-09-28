"""Compose mixed projection ports with the actual retained P31 stage programs.

This admits complete addresses/fixed routes/program storage. Root consumers and
automatic mixed-stage execution remain open; no host driver may call it a layer.
"""
import hashlib
import json
import re
from pathlib import Path
from spatial.layer_mlp_network import emit_routes
from spatial.layer_schedule import xy_rank
from spatial.mixer_projection import lower_mixer_projections,FIELDS


def compose_mixer(files,width,height,profiles,stage,network,device_control=False,routed_control=False):
    root=Path(__file__).resolve().parents[1]
    region=next(r for r in stage['regions'] if r['role']=='mix')
    plan=lower_mixer_projections(region,stage['request_controller']['pe'])
    workers={tuple(w['pe']):w for w in plan['workers']}
    wrapper=(root/'runtime/mixer_projection_port.cslpart').read_bytes()
    for name in ('mixer_native','mixer_projection','mixer_operand','bf16_row'):
        files[name+'.csl']=(root/'csl'/ (name+'.csl')).read_bytes()
    bindings={};setups=[]
    for profile in profiles:
        pe=tuple(profile['pe'])
        if pe not in workers or list(pe)==stage['request_controller']['pe']:continue
        source=profile['source'];destination='mixer_'+source
        if source not in ('layer_norm_bridge.csl','layer_norm_sender.csl','layer_mlp_standby.csl'):
            raise ValueError('Unexpected mixer cohost role')
        # These programs share arithmetic DSRs/microthreads across phases.
        # Reject an overlapping norm/mixer invocation before arming any DMA.
        guarded,n=re.subn(rb'(fn arm\([^)]*\) void \{)',rb'\1@assert(mixer.drained() and mixer_status[2]==0);',files[source])
        if n!=1:raise ValueError('Expected exactly one original cohost arm entry')
        idle=b'@assert(!active);' if source!='layer_mlp_standby.csl' else b''
        files[destination]=guarded+b'\n'+wrapper.replace(b'MIXER_COHOST_IDLE',idle)
        bindings[destination]=dict(original=source,original_sha256=hashlib.sha256(files[source]).hexdigest(),
                                   composition_sha256=hashlib.sha256(wrapper).hexdigest(),guards='Both entry directions require the other program drained')
        profile['source']=destination
        rank=xy_rank(region,pe)
        profile['parameters'].update(mixer_rank_parity=rank%2,mixer_input48_queue=4 if source=='layer_norm_sender.csl' else 5,
                                     mixer_can_root=rank%40==0 or rank%48==0)
        setups.append([list(pe),[d[k] for d in workers[pe]['descriptors'] for k in FIELDS]])
    # The live source port borrows the idle controller's existing arenas. It
    # cannot overlap a norm/MLP epoch, and adds no per-worker quantizer copy.
    controller=next(p for p in profiles if p['pe']==stage['request_controller']['pe'])
    original=files[controller['source']]
    guarded,n=re.subn(rb'(fn arm\([^)]*\) void \{)',rb'\1@assert(!mixer_operand_busy);',original)
    if n!=1:raise ValueError('Expected one controller arm entry')
    producer=(root/'runtime/mixer_operand_producer.cslpart').read_bytes()
    producer=producer.replace(b'MIXER_OPERAND_COMPLETION',b'device_operand_complete();' if routed_control else b'')
    legacy=b'fn mixer_send_operand(invocation:u32,group:u16,parts:u16) void {mixer_emit_operand(mip,invocation,group,parts);sys.unblock_cmd_stream();}'
    producer=producer.replace(b'MIXER_OPERAND_LEGACY_ENTRY',b'' if routed_control else legacy)
    producer=producer.replace(b'MIXER_OPERAND_LEGACY_EXPORT',b'' if routed_control else b'@export_symbol(mixer_send_operand);')
    controller['source']='mixer_layer_mlp_controller.csl';files[controller['source']]=guarded+b'\n'+producer
    bindings[controller['source']]=dict(original='layer_mlp_controller.csl',original_sha256=hashlib.sha256(original).hexdigest(),
                                       composition_sha256=hashlib.sha256(producer).hexdigest(),guards='Norm/MLP arm and mixer sender require the other phase idle')
    routes=network['routes']+plan['input_routes']
    routes += [dict(pe=r['pe'],color=r['color'],rx=[r['rx']],tx=[r['tx']]) for r in plan['reduction_routes']]
    if routed_control:
        if not device_control:raise ValueError('Routed control requires internal device ports')
        from spatial.device_network import lower_device_network
        control=lower_device_network(region['rect'])
        if control['gateway']!=stage['request_controller']['pe']:raise ValueError('Device gateway controller ownership')
        routes+=control['routes'];directions={tuple(w['pe']):w for w in control['workers']}
        files['device-network.json']=(json.dumps(control,separators=(',',':'))+'\n').encode()
        files['device_network.py']=(root/'spatial/device_network.py').read_bytes()
        gateway=(root/'runtime/device_gateway.cslpart').read_bytes().replace(b'DEVICE_TARGET_COUNT',str(len(control['workers'])).encode())
        current=files[controller['source']]
        sdk=b'const sys=@import_module("<memcpy/memcpy>",memcpy_params);'
        if current.count(sdk)!=1:raise ValueError('Expected one SDK controller import')
        fields=['first_pe','last_pe','MEMCPYH2D_1','MEMCPYH2D_2','MEMCPYH2D_3','MEMCPYH2D_4','MEMCPYD2H_1','MEMCPYD2H_2','MEMCPYD2H_3','MEMCPYD2H_4','LAUNCH']
        if any(r['pe']==stage['request_controller']['pe'] and r['color'] in (14,15) for r in routes):raise ValueError('SDK gateway stream color aliases neural/controller route')
        overrides={'MEMCPYH2D_1':'14','MEMCPYD2H_1':'15'}
        params=','.join('.'+f+'='+overrides.get(f,'memcpy_params.'+f) for f in fields)
        current=current.replace(sdk,('const sys=@import_module("<memcpy/memcpy>",.{'+params+'});').encode())
        for entry in ('arm','mixer_emit_operand'):
            current,n=re.subn(rb'(fn '+entry.encode()+rb'\([^)]*\) void \{)',rb'\1@assert(!device_gateway_active and device_submit_cursor==0);',current)
            if n!=1:raise ValueError('Expected one guarded controller entry')
        files[controller['source']]=current+b'\n'+gateway
    keys=[(*r['pe'],r['color']) for r in routes]
    if len(set(keys))!=len(keys):raise ValueError('Mixer fixed colors alias the existing MLP/norm graph')
    if device_control:
        from spatial.device_control import adapt_mixer
        adaptations={}
        for profile in profiles:
            source=profile['source']
            if source not in ('mixer_layer_norm_bridge.csl','mixer_layer_norm_sender.csl','mixer_layer_mlp_standby.csl'):continue
            adapted='device_'+source
            if adapted not in files:files[adapted],adaptations[adapted]=adapt_mixer(files[source],source,forward_host_commands=True,routed=routed_control)
            profile['source']=adapted
            if routed_control:
                endpoint=directions[tuple(profile['pe'])]
                profile['parameters'].update(control_reply_rx=endpoint['reply_rx'],control_reply_tx=endpoint['reply_tx'])
        files['device_control.csl']=(root/'csl/device_control.csl').read_bytes()
        files['device_control.py']=(root/'spatial/device_control.py').read_bytes()
        files['device-adaptation.json']=(json.dumps(adaptations,indent=2)+'\n').encode()
        files['device-control-admission.json']=(json.dumps(dict(compiled_ports_only=not routed_control,complete_control_routes=routed_control,
            host_command_forwarding=True,host_command_forwarding_executed=False,
            scope='Addressed gateway/return routing is source-qualified only; complete-stage compile and control/neural execution are separately qualified.' if routed_control else 'Internal control ports are compiled but gateway command/response paths are not connected. No load or complete-stage execution admission.'),indent=2)+'\n').encode()
    ox,oy=stage['rect'][:2];classes={}
    for p in profiles:
        key=(p['source'],json.dumps(p['parameters'],sort_keys=True))
        classes.setdefault(key,dict(source=p['source'],parameters=p['parameters'],pes=[]))['pes'].append(p['pe'])
    stream=',.MEMCPYH2D_1=MEMCPYH2D_DATA_1_ID,.MEMCPYD2H_1=MEMCPYD2H_DATA_1_ID' if routed_control else ''
    layout=['param MEMCPYH2D_DATA_1_ID:i16=14;param MEMCPYD2H_DATA_1_ID:i16=15;'] if routed_control else []
    layout+=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}{stream}}});']
    if routed_control:
        # Register the host streaming color in layout metadata, but install it
        # only on the gateway. Other SDK kernels keep their neural C14 routes.
        layout+=['fn ordinary_memcpy_params(x:i16) memcpy.Params {var p=memcpy.get_params(x);p.MEMCPYH2D_1=-1;p.MEMCPYD2H_1=-1;return p;}']
    layout+=['layout {',f' @set_rectangle({width},{height});']
    if routed_control:layout+=[' @comptime_assert(MEMCPYH2D_DATA_1_ID==14 and MEMCPYD2H_DATA_1_ID==15);']
    for i,c in enumerate(classes.values()):
        coords=[v for x,y in c['pes'] for v in (x-ox,y-oy)];name=f'program_{i}'
        layout.append(' const %s=[%d]u16{%s};'%(name,len(coords),','.join(map(str,coords))))
        params=','.join('.%s=%s'%(k,v if isinstance(v,str) else str(v).lower()) for k,v in c['parameters'].items())
        provider='ordinary_memcpy_params' if routed_control else 'memcpy.get_params'
        layout.append(' for(@range(u16,%d))|i|{@set_tile_code(%s[2*i],%s[2*i+1],"%s",.{.memcpy_params=%s(%s[2*i]),%s});}'%(len(coords)//2,name,name,c['source'],provider,name,params))
    layout.append(emit_routes(dict(rect=stage['rect'],routes=routes)))
    declarations=[line for line in files['layout.csl'].decode().splitlines() if line.strip().startswith('@export_name')]
    for name,typ in [('mixer_setup','u16'),('mixer_status','u32'),('mixer_audit','u32'),('mixer_output','u32'),('mixer_operand_input','u32'),('mixer_operand_status','u32')]:
        declarations.append(f' @export_name("{name}",[*]{typ},false);')
    declarations += [' @export_name("mixer_begin",fn(u32,u32,u16)void);',' @export_name("mixer_consume",fn(u32,u16)void);']
    if not routed_control:declarations.append(' @export_name("mixer_send_operand",fn(u32,u16,u16)void);')
    if routed_control:
        declarations+=[' @export_name("%s",[*]u32,false);'%n for n in ['device_command','device_response','device_audit']]
    if device_control:
        exported=set()
        for source in {p['source'] for p in profiles}:
            for value in re.findall(r'@export_symbol\(([^;]+)\);',files[source].decode()):
                exported.add(value.split(',')[-1].strip().strip('"'))
        declarations=[line for line in declarations if re.search(r'@export_name\("([^"]+)"',line).group(1) in exported]
    layout+=declarations+['}']
    files['layout.csl']=('\n'.join(layout)+'\n').encode()
    scope='Original mixer projection cohosts with one shared original BF16/FP8 operand producer and fixed paths. Root conv/gate/state/residual consumers and complete-stage execution remain unconnected; compile admission only.'
    files['profiles.json']=(json.dumps(dict(profiles=list(classes.values()),application=[width,height],physical=False,executed=False,whole_stage=stage['id'],scope=scope),separators=(',',':'))+'\n').encode()
    files['mixer-setups.json']=(json.dumps(setups,separators=(',',':'))+'\n').encode()
    files['mixer-projections.json']=(json.dumps(plan,separators=(',',':'))+'\n').encode()
    files['mixer-composition.json']=(json.dumps(dict(bindings=bindings,wrapper=wrapper.decode(),producer_wrapper=producer.decode(),source_only=True,scope=scope),indent=2)+'\n').encode()
    for name in ('mixer_composition','mixer_projection'):
        files[name+'.py']=(root/'spatial'/(name+'.py')).read_bytes()
    return files,width,height,profiles
