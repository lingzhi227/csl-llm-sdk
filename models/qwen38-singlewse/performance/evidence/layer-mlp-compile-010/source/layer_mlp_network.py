"""Connect the original resident gate/up, group quantization and down projection.

Color2 is a tagged native-input broadcast,18 an exclusively granted return bus,
20 a filtered grant broadcast (available in the same SDK's qualified P15 kernel).
The arbiter receives the entire frame before granting another source. A source
selects RAMP only under its grant and restores its single transit RX after an
output-queue flush; WSE-3 forbids simultaneous local/transit inputs.
"""
from collections import Counter
from copy import deepcopy
from spatial.layer_routes import direction, worker, edge_color
from spatial.layer_schedule import rank_xy
from spatial.layer_fused_routes import (COLORS, compact_tree_groups, reduction_routes,
                                       shortest_route, lower_path, root_parameters, actor_parameters)


def input_selectors(region,shared_inputs=False):
    groups=list(compact_tree_groups(region)); blocks=region['matrices'][0]['k_blocks']
    tail=[g for g in groups if g['workers']!=blocks]
    if len(tail)!=1:raise ValueError('Exactly one original short native group required')
    full_base=0 if region['role']=='gate_up' else 1024; tail_base=full_base+512
    tail_workers=tail[0]['workers'];stride=(blocks+tail_workers-1)//tail_workers
    selectors={};bindings=[]
    for group in groups:
        for node in group['nodes']:
            w=worker(region,group,node);tag=(full_base if group['workers']==blocks else tail_base)+node['rank']
            if shared_inputs:
                first=node['rank']
                tag=full_base+(first%tail_workers)*stride+first//tail_workers
                maximum=tag if group['workers']==blocks else tag+node['parts']-1
                selectors[tuple(w['pe'])]=(tag,maximum)
                bindings.append(dict(pe=w['pe'],tag=tag,max_tag=maximum,slices=w['native_input_slices'],parts=node['parts']))
            else:
                selectors[tuple(w['pe'])]=tag
                bindings.append(dict(pe=w['pe'],tag=tag,slices=w['native_input_slices'],parts=node['parts']))
    frames=[]
    for k in range(blocks):
        if shared_inputs:frames.append(dict(k=k,tag=full_base+(k%tail_workers)*stride+k//tail_workers,part=k//tail_workers))
        else:frames.extend([dict(k=k,tag=full_base+k,part=0),
                            dict(k=k,tag=tail_base+k%tail_workers,part=k//tail_workers)])
    # Independent original-K coverage, including different ragged-tail periods.
    for b in bindings:
        observed=[(0 if shared_inputs and b['parts']==1 else f['part'],f['k']) for f in frames if b['tag']<=f['tag']<=b.get('max_tag',b['tag'])]
        if sorted(observed)!=list(enumerate(b['slices'])):raise ValueError('Tagged input changes original native K ownership')
    return dict(selectors=selectors,bindings=bindings,frames=frames,tail_workers=tail_workers)


def broadcast(source, receivers, color):
    """Bottom horizontal trunk, parallel north-going columns, filtered RAMPs."""
    cx,cy=source;top={}
    for x,y in receivers:
        if y>cy:raise ValueError('Controller must be on/below all recipients')
        top[x]=min(top.get(x,cy),y)
    minimum=min(cx,*top);maximum=max(cx,*top);result=[]
    for x in range(minimum,maximum+1):
        tx=[]
        if x<=cx and x>minimum:tx.append('WEST')
        if x>=cx and x<maximum:tx.append('EAST')
        if x in top and top[x]<cy:tx.append('NORTH')
        if (x,cy) in receivers:tx.append('RAMP')
        record=dict(pe=[x,cy],color=color,rx=['RAMP' if x==cx else 'EAST' if x<cx else 'WEST'],tx=tx)
        if (x,cy) in receivers:
            value=receivers[x,cy];record['filter_tag']=value[0] if isinstance(value,tuple) else value
            if isinstance(value,tuple):record['filter_max']=value[1]
        result.append(record)
        if x not in top:continue
        for y in range(cy-1,top[x]-1,-1):
            tx=['NORTH'] if y>top[x] else []
            if (x,y) in receivers:tx.append('RAMP')
            record=dict(pe=[x,y],color=color,rx=['SOUTH'],tx=tx)
            if (x,y) in receivers:
                value=receivers[x,y];record['filter_tag']=value[0] if isinstance(value,tuple) else value
                if isinstance(value,tuple):record['filter_max']=value[1]
            result.append(record)
    return result


def granted_bus(stage, senders):
    controller=stage['request_controller']['pe'];cx,cy=controller
    gate=next(r for r in stage['regions'] if r['role']=='gate_up')
    y=gate['rect'][1]+gate['rect'][3]-1;right=stage['rect'][0]+stage['rect'][2]-1
    xs=[x for x,yy in senders if yy==y]
    path=[(x,y) for x in range(min(xs),right+1)]
    path += [(right,yy) for yy in range(y+1,cy+1)]
    if any(yy==cy-1 for _,yy in senders):
        # Extend the single-input line through the new mixer-row norm owners.
        # The old down sinks end east of the controller. Turn north one column
        # before it, visit the norm row, then return on the unused west tail.
        left=min(x for x,yy in senders if yy==cy-1)
        path += [(x,cy) for x in range(right-1,cx,-1)]
        path += [(cx+1,cy-1)]
        path += [(x,cy-1) for x in range(cx,left-1,-1)]
        path += [(x,cy) for x in range(left,cx+1)]
    else:path += [(x,cy) for x in range(right-1,cx-1,-1)]
    if not set(senders)<=set(path):raise ValueError('Every granted source must lie on the directed bus')
    if len(set(path))!=len(path):raise ValueError('Return bus cycles')
    result=[]
    for i,pe in enumerate(path):
        rx=['RAMP' if i==0 else direction(pe,path[i-1])]
        result.append(dict(pe=list(pe),color=18,rx=rx,tx=['RAMP' if i+1==len(path) else direction(pe,path[i+1])],
                           exclusive_grant_bus=True,grant_selected_injection=pe in senders))
    return result


def group_packets(schedule,gate_tail,down_tail,shared_inputs=False):
    """Coalesce one received128-value group without changing its wire sequence."""
    result=[]
    for sequence,g in enumerate(schedule):
        if g['kind']>=2:continue
        columns=32 if g['kind']==0 else 64;tail=gate_tail if g['kind']==0 else down_tail
        base=0 if g['kind']==0 else 1024;slices=[];blocks=160 if g['kind']==0 else 272
        stride=(blocks+tail-1)//tail
        for segment in range(128//columns):
            k=g['index']*(128//columns)+segment
            for alias in range(1 if shared_inputs else 2):
                slices.append(dict(k=k,tag=base+(k%tail)*stride+k//tail if shared_inputs else base+k if alias==0 else base+512+k%tail,
                    part=k//tail if shared_inputs else 0 if alias==0 else k//tail,segment=segment,offset=len(slices)*(columns+5)))
        result.append(dict(sequence=sequence,kind=g['kind'],group=g['index'],columns=columns,
                           words=len(slices)*(columns+5),slices=slices))
    return result


def down_sinks(region):
    groups=list(compact_tree_groups(region));roots={tuple(g['root_pe']) for g in groups}
    _,y,width,height=region['rect'];bottom=y+height-1;candidates=[]
    for group in groups:
        for node in group['nodes']:
            w=worker(region,group,node)
            if w['pe'][1]==bottom and tuple(w['pe']) not in roots and w['parameters']['bank_words']*4<=28000:
                candidates.append(w)
    candidates.sort(key=lambda w:tuple(w['pe']));ordered=sorted(groups,key=lambda g:tuple(g['root_pe']))
    dp={(0,j):(0,[]) for j in range(len(candidates)+1)}
    for i in range(1,len(groups)+1):
        for j in range(i,len(candidates)+1):
            distance=sum(abs(a-b) for a,b in zip(ordered[i-1]['root_pe'],candidates[j-1]['pe']))
            take=(dp[i-1,j-1][0]+distance,dp[i-1,j-1][1]+[j-1])
            dp[i,j]=min(take,dp.get((i,j-1),(10**9,[])))
    if (len(groups),len(candidates)) not in dp:raise ValueError('No original low-payload output sinks')
    selected={g['index']:candidates[j] for g,j in zip(ordered,dp[len(groups),len(candidates)][1])}
    sinks=[]
    for g in groups:
        w=selected[g['index']]
        sinks.append(dict(root=g['index'],pe=w['pe'],rank=w['rank'],payload=w['parameters']['bank_words']*4,
                          first_row=g['output_start']*4,rows=g['output_count']*4))
    trees=[];used={c:set() for c in COLORS}
    for g in groups:
        for tree in reduction_routes(region,g):
            for r in tree['routes']:
                r=dict(**r,color=tree['color']);r['rx']=[r['rx']];r['tx']=[r['tx']]
                if tuple(r['pe']) in used[r['color']]:raise ValueError('Down tree alias')
                used[r['color']].add(tuple(r['pe']));trees.append(r)
    endpoints={tuple(g['root_pe']) for g in groups}|{tuple(s['pe']) for s in sinks}
    pressure=Counter(pe for positions in used.values() for pe in positions);flows=[]
    for g in sorted(groups,key=lambda g:-sum(abs(a-b) for a,b in zip(g['root_pe'],sinks[g['index']]['pe']))):
        choices=[];sink=sinks[g['index']]
        for color in COLORS:
            path=shortest_route(g['root_pe'],sink['pe'],color,region['rect'],used,pressure,endpoints)
            if path:choices.append((sum(1+pressure[p[:2]]**2 for p in path),color,path))
        if not choices:raise ValueError('Down projection path unavailable')
        *_,path=min(choices)
        for x,y,c in path:used[c].add((x,y));pressure[x,y]+=1
        routes=lower_path(path)
        for r in routes:r['rx']=[r['rx']]
        flows.append(dict(root=g['index'],source=g['root_pe'],destination=sink['pe'],routes=routes))
    return groups,sinks,trees,flows


def build_mlp_network(stage, gate_routes,shared_inputs=False,norm_bridge=False):
    regions={r['role']:r for r in stage['regions']};gate=regions['gate_up'];down=regions['down']
    if gate_routes['region']!=gate['id'] or not gate_routes.get('copied_transport'):raise ValueError('Current exact P25 copy routes required')
    gg=list(compact_tree_groups(gate));dg,sinks,down_trees,down_flows=down_sinks(down)
    # Reuse all33 fusion PEs; seven additional original low-payload PEs supply
    # the remaining input quantization groups. Arithmetic runs spatially.
    chosen={a['rank']:dict(rank=a['rank'],pe=a['pe'],fusion_producer=a['producer']) for a in gate_routes['actors']}
    roots={tuple(g['root_pe']) for g in gg};bottom=gate['rect'][1]+gate['rect'][3]-1
    for candidate in sorted(gate['quantization_actors'],key=lambda a:tuple(a['pe'])):
        if len(chosen)>=40:break
        if candidate['pe'][1]==bottom and tuple(candidate['pe']) not in roots:
            chosen.setdefault(candidate['rank'],dict(rank=candidate['rank'],pe=candidate['pe'],fusion_producer=None))
    if len(chosen)!=40:raise ValueError('Forty spatial original group128 quantizers required')
    senders=sorted(chosen.values(),key=lambda a:tuple(a['pe']))
    for i,s in enumerate(senders):s.update(id=i,input_group=i)
    if norm_bridge:
        mx,my,mw,mh=regions['mix']['rect']
        if mw<42 or stage['request_controller']['pe']!=[mx+mw-1,my+mh-1]:
            raise ValueError('Norm row requires admitted layer00-compatible mixer geometry')
        fusion_senders=[dict(id=40+i,pe=a['pe'],rank=a['rank'],fusion_producer=a['producer'])
                        for i,a in enumerate(gate_routes['actors'])]
        senders=[dict(id=i,input_group=i,pe=[mx+i,my+mh-2],norm_bridge=True) for i in range(40)]+fusion_senders
    for i,sink in enumerate(sinks):senders.append(dict(id=len(senders),pe=sink['pe'],rank=sink['rank'],output_sink=i))
    ginput=input_selectors(gate,shared_inputs);dinput=input_selectors(down,shared_inputs)
    inputs={**ginput['selectors'],**dinput['selectors']}
    grants={tuple(s['pe']):s['id'] for s in senders}
    routes=[]
    for flow in gate_routes['reduction_trees']+gate_routes['flows']:
        for original in flow['routes']:
            r=deepcopy(original);r['rx']=[r['rx']]
            if isinstance(r['tx'],str):r['tx']=[r['tx']]
            routes.append(r)
    routes += down_trees+[r for f in down_flows for r in f['routes']]
    routes += broadcast(stage['request_controller']['pe'],inputs,2)
    routes += broadcast(stage['request_controller']['pe'],grants,20)
    routes += granted_bus(stage,set(grants))
    if norm_bridge:
        norms=[s for s in senders if s.get('norm_bridge')]
        routes += broadcast(stage['request_controller']['pe'],{tuple(s['pe']):s['input_group'] for s in norms},7)
        for i,owner in enumerate(norms):
            if i<39:routes.append(dict(pe=owner['pe'],color=4+(i+1)%2,rx=['EAST'],tx=['RAMP']))
            if i>0:routes.append(dict(pe=owner['pe'],color=4+i%2,rx=['RAMP'],tx=['WEST']))
            routes.append(dict(pe=owner['pe'],color=6,rx=['RAMP' if i==0 else 'WEST'],
                               tx=(['EAST'] if i<39 else [])+(['RAMP'] if i>0 else [])))
    schedule=[dict(target=s['id'],kind=0,index=s['input_group'],words=67) for s in senders if 'input_group' in s]
    # These commands start independent quantizers but grant no return-bus
    # injection. FIFO order on each recipient puts prepare before frame fetch.
    prepare_schedule=[dict(target=s['id'],kind=3,index=s['input_group'],words=0) for s in senders if 'input_group' in s]
    by_producer={s['fusion_producer']:s for s in senders if s.get('fusion_producer') is not None}
    for turn in range(max(len(a['groups']) for a in gate_routes['actors'])):
        for a in gate_routes['actors']:
            if turn<len(a['groups']):schedule.append(dict(target=by_producer[a['producer']]['id'],kind=1,index=a['groups'][turn],words=67))
    schedule += [dict(target=40+s['root'],kind=2,index=s['root'],words=3+s['rows']//2,first_row=s['first_row'],rows=s['rows']) for s in sinks]
    if norm_bridge:schedule += [dict(target=s['id'],kind=4,index=s['input_group'],words=67,
                                             first_row=s['input_group']*128,rows=128) for s in senders if s.get('norm_bridge')]
    packets=group_packets(schedule,ginput['tail_workers'],dinput['tail_workers'],shared_inputs)
    result=dict(stage=stage['id'],rect=stage['rect'],controller=stage['request_controller']['pe'],routes=routes,
                senders=senders,down_sinks=sinks,down_output_flows=down_flows,grant_schedule=schedule,prepare_schedule=prepare_schedule,
                distribution_packets=packets,shared_inputs=shared_inputs,norm_bridge=norm_bridge,controller_transport=dict(frame_words=max(g['words'] for g in schedule),
                    packet_words=max(p['words'] for p in packets),response_capacity=1,packet_capacity=1,
                    buffer_release='Receive arena reusable only after full frame consumption; packet immutable until output DMA completion.',
                    prefetch='At most one next exclusive response while a distinct native packet is in flight.'),
                input_bindings=ginput['bindings']+dinput['bindings'],gate_tail_workers=ginput['tail_workers'],
                down_tail_workers=dinput['tail_workers'],gate_wire_words=37,down_wire_words=69,
                exclusive_bus_rule='A grant follows complete prior-frame receipt. Its sender selects single RAMP RX, then qflush restores single transit RX before another local send. Every OQ retains its fixed color.',
                full_model=False,executed=False,physical=False)
    result['audit']=audit_mlp_network(stage,result)
    return result


def audit_mlp_network(stage, result):
    """Walk every input/grant broadcast and each granted source to the arbiter."""
    installed={};x,y,w,h=stage['rect'];delta={'EAST':(1,0),'WEST':(-1,0),'SOUTH':(0,1),'NORTH':(0,-1)}
    opposite={'EAST':'WEST','WEST':'EAST','SOUTH':'NORTH','NORTH':'SOUTH'}
    for r in result['routes']:
        key=(*r['pe'],r['color'])
        if key in installed:raise ValueError('Composed stage PE/color alias')
        if not x<=key[0]<x+w or not y<=key[1]<y+h or r['color']>=21:raise ValueError('Stage boundary or SDK color violation')
        if len(r['rx'])!=1:raise ValueError('WSE-3 requires exactly one static receive direction')
        if r.get('color_swap_x') or r.get('color_swap_y'):raise ValueError('Unsupported WSE-3 color swap')
        installed[key]=r
    def follow(source,color,tag=None):
        todo=[(tuple(source),'RAMP')];seen=set();sinks=[]
        while todo:
            pe,incoming=todo.pop()
            if pe in seen:raise ValueError('Physical path cycle or duplicated broadcast')
            seen.add(pe);r=installed[(*pe,color)]
            if incoming not in r['rx'] and not (pe==tuple(source) and incoming=='RAMP' and r.get('grant_selected_injection')):raise ValueError('Disconnected physical route')
            for tx in r['tx']:
                if tx=='RAMP':
                    if 'filter_tag' not in r or r['filter_tag']<=tag<=r.get('filter_max',r['filter_tag']):sinks.append(pe)
                else:
                    dx,dy=delta[tx];todo.append(((pe[0]+dx,pe[1]+dy),opposite[tx]))
        return sinks
    by_tag={}
    shared=result.get('shared_inputs',False)
    for b in result['input_bindings']:
        for tag in range(b['tag'],b.get('max_tag',b['tag'])+1):by_tag.setdefault(tag,set()).add(tuple(b['pe']))
    # Reconstruct expected native K ownership independently of claimed bindings.
    expected_bindings=[]
    for region in stage['regions']:
        if region['role'] in ('gate_up','down'):expected_bindings+=input_selectors(region,shared)['bindings']
    if result['input_bindings']!=expected_bindings:raise ValueError('Input binding changes original native ownership')
    for tag,expected in by_tag.items():
        received=follow(result['controller'],2,tag)
        if set(received)!=expected or len(received)!=len(expected):raise ValueError('Input filter selects wrong original K owners')
    for sender in result['senders']:
        if follow(result['controller'],20,sender['id'])!=[tuple(sender['pe'])]:raise ValueError('Grant target ambiguity')
        if follow(sender['pe'],18)!=[tuple(result['controller'])]:raise ValueError('Granted frame does not reach arbiter')
    if result.get('norm_bridge'):
        norms=[s for s in result['senders'] if s.get('norm_bridge')]
        mx,my,mw,mh=next(r['rect'] for r in stage['regions'] if r['role']=='mix')
        if [(s['id'],s['input_group'],s['pe']) for s in norms]!=[(i,i,[mx+i,my+mh-2]) for i in range(40)]:
            raise ValueError('Norm residual/gain chunk ownership')
        for i,s in enumerate(norms):
            if follow(result['controller'],7,i)!=[tuple(s['pe'])]:raise ValueError('Down residual chunk route')
            if i and follow(s['pe'],4+i%2)!=[tuple(norms[i-1]['pe'])]:raise ValueError('Norm reduction route')
        if set(follow(norms[0]['pe'],6))!={tuple(s['pe']) for s in norms[1:]}:raise ValueError('RMS inverse broadcast')
        expected_final=[dict(target=s['id'],kind=4,index=s['input_group'],words=67,
                             first_row=s['input_group']*128,rows=128) for s in norms]
        if [g for g in result['grant_schedule'] if g['kind']==4]!=expected_final:raise ValueError('Successor RMS collection')
    for flow in result['down_output_flows']:
        if follow(flow['source'],flow['routes'][0]['color'])!=[tuple(flow['destination'])]:raise ValueError('Down output route does not reach its sink')
    for region in stage['regions']:
        if region['role'] not in ('gate_up','down'):continue
        for group in compact_tree_groups(region):
            for node in group['nodes']:
                if node['parent'] is None:continue
                source=rank_xy(region,group['rank_start']+node['rank'])
                destination=rank_xy(region,group['rank_start']+node['parent'])
                if follow(source,edge_color(node))!=[tuple(destination)]:raise ValueError('Disconnected original native reduction')
    phases=Counter(s['kind'] for s in result['grant_schedule'])
    if phases!=({0:40,1:136,2:10,4:40} if result.get('norm_bridge') else {0:40,1:136,2:10}):raise ValueError('Incomplete original MLP grant schedule')
    expected=[dict(target=s['id'],kind=3,index=s['input_group'],words=0) for s in result['senders'] if 'input_group' in s]
    if result['prepare_schedule']!=expected or len(expected)!=40:raise ValueError('Missing, duplicate or injecting input preparation')
    if {(p['target'],p['index']) for p in expected}!={(g['target'],g['index']) for g in result['grant_schedule'] if g['kind']==0}:
        raise ValueError('Input preparation/fetch ownership mismatch')
    if sorted(s['index'] for s in result['grant_schedule'] if s['kind']==1)!=list(range(136)):raise ValueError('Missing original activation group')
    packets=result['distribution_packets']
    expected_sequence=[i for i,g in enumerate(result['grant_schedule']) if g['kind']<2]
    if [p['sequence'] for p in packets]!=expected_sequence:raise ValueError('Packet order differs from response order')
    for p in packets:
        g=result['grant_schedule'][p['sequence']];cols=32 if g['kind']==0 else 64
        if (p['kind'],p['group'],p['columns'])!=(g['kind'],g['index'],cols):raise ValueError('Packet owner differs from original group')
        aliases=1 if shared else 2
        if len(p['slices'])!=aliases*128//cols or p['words']!=len(p['slices'])*(cols+5):raise ValueError('Coalesced packet extent')
        for i,s in enumerate(p['slices']):
            if s['offset']!=i*(cols+5) or s['segment']!=i//aliases or s['k']!=g['index']*(128//cols)+i//aliases:raise ValueError('Overlapping or reordered slice')
    for kind,role in [(0,'gate_up'),(1,'down')]:
        region=next(r for r in stage['regions'] if r['role']==role)
        expected=Counter((f['k'],f['tag'],f['part']) for f in input_selectors(region,shared)['frames'])
        actual=Counter((s['k'],s['tag'],s['part']) for p in packets if p['kind']==kind for s in p['slices'])
        if actual!=expected:raise ValueError('Coalescing changes original native operand routing')
    if result['controller_transport']['frame_words']!=273 or result['controller_transport']['packet_words']!=(148 if shared else 296):
        raise ValueError('Controller arena geometry')
    if result['controller_transport']['response_capacity']!=1 or result['controller_transport']['packet_capacity']!=1:
        raise ValueError('Controller lifetime proof only admits one response and one packet')
    rows=[i for s in result['down_sinks'] for i in range(s['first_row'],s['first_row']+s['rows'])]
    if sorted(rows)!=list(range(5120)):raise ValueError('Missing or duplicate original down output')
    return dict(passed=True,original_gate_rows=17408,original_down_rows=5120,native_workers=len(result['input_bindings']),
                input_tags=len(by_tag),spatial_input_quantizers=40,fusion_actors=33,output_sinks=10,
                grants=len(result['grant_schedule']),preparation_commands=40,distribution_packets=len(packets),
                distribution_wire_words=sum(p['words'] for p in packets),route_entries=len(installed),bus_sources=len(result['senders']),
                statically_routed=True,exclusive_bus_protocol_executed=False,numerical_qualified=False,physical=False)


def emit_routes(result):
    """Group identical static router configurations, retaining every coordinate."""
    ox,oy=result['rect'][:2];classes={};lines=[]
    for r in result['routes']:
        key=(r['color'],tuple(r['rx']),tuple(r['tx']),'filter_tag' in r,'filter_max' in r)
        classes.setdefault(key,[]).extend([r['pe'][0]-ox,r['pe'][1]-oy]+([r['filter_tag']] if key[3] else [])+([r['filter_max']] if key[4] else []))
    for n,((color,rx,tx,filtered,ranged),coords) in enumerate(classes.items()):
        stride=4 if ranged else 3 if filtered else 2;name=f'route_{n}'
        lines.append(' const %s=[%d]u16{%s};'%(name,len(coords),','.join(map(str,coords))))
        fields='.routes=.{.rx=.{%s},.tx=.{%s}}'%(','.join(rx),','.join(tx))
        if filtered:fields+=',.filter=.{.kind=.{.range=true},.min_idx=%s[%d*i+2],.max_idx=%s[%d*i+%d]}'%(name,stride,name,stride,3 if ranged else 2)
        lines.append(' for(@range(u16,%d))|i|{@set_color_config(%s[%d*i],%s[%d*i+1],@get_color(%d),.{%s});}'%(len(coords)//stride,name,stride,name,stride,color,fields))
    return '\n'.join(lines)+'\n'


def worker_profiles(stage, gate_routes, network):
    """Bind every queue to the audited fabric without changing original banks."""
    profiles={};regions={r['role']:r for r in stage['regions']}
    for role in ('gate_up','down'):
        region=regions[role];groups=list(compact_tree_groups(region))
        for group in groups:
            for node in group['nodes']:
                w=worker(region,group,node);p=w['parameters'];p.update(copy_transport=True,tagged_input=True,mlp_component=True)
                if network.get('shared_inputs'):p['shared_inputs']=True
                if role=='gate_up' and p['root']:p.update(root_parameters(gate_routes,group))
                if role=='down' and p['root']:
                    flow=next(f for f in network['down_output_flows'] if f['root']==group['index'])
                    p['out_color']=flow['routes'][0]['color']
                w['source']='layer_projection.csl';profiles[tuple(w['pe'])]=w
        if role=='gate_up':
            for actor in gate_routes['actors']:
                w=profiles[tuple(actor['pe'])];w['parameters'].update(actor_parameters(gate_routes,groups,actor))
    for sender in network['senders']:
        if sender.get('norm_bridge'):continue
        p=profiles[tuple(sender['pe'])]['parameters']
        bus=next(r for r in network['routes'] if r['color']==18 and r['pe']==sender['pe'])
        p.update(mlp_sender=True,sender_id=sender['id'],bus_rx=bus['rx'][0],bus_tx=bus['tx'][0])
        if 'input_group' in sender:p.update(input_quantizer=True,input_group=sender['input_group'])
        elif 'output_sink' in sender:
            sink=network['down_sinks'][sender['output_sink']]
            flow=next(f for f in network['down_output_flows'] if f['root']==sink['root'])
            p.update(output_sink=True,sink_color=flow['routes'][-1]['color'],sink_index=sink['root'],sink_first=sink['first_row'],sink_rows=sink['rows'])
        if p['root']:raise ValueError('Bus sender/root output queue collision')
    installed={(*r['pe'],r['color']):r for r in network['routes']}
    for pe,w in profiles.items():
        p=w['parameters'];incoming=[2];outgoing=[p['out_color']]
        if p['children']:incoming.append(p['left_color'])
        if p['children']>1:incoming.append(p['right_color'])
        if p.get('boundary_rows'):outgoing.append(p['boundary_out_color'])
        if p.get('fusion_actor'):
            incoming.append(p['actor_main_color'])
            if p['actor_has_suffix']:incoming.append(p['actor_suffix_color'])
        if p.get('mlp_sender'):incoming.append(20);outgoing.append(18)
        if p.get('output_sink'):incoming.append(p['sink_color'])
        if len(set(incoming))!=len(incoming) or len(set(outgoing))!=len(outgoing):raise ValueError('Queue color alias')
        if any('RAMP' not in installed[(*pe,c)]['tx'] for c in incoming):raise ValueError('Missing input port')
        if any('RAMP' not in installed[(*pe,c)]['rx'] and not installed[(*pe,c)].get('grant_selected_injection') for c in outgoing):raise ValueError('Missing output port')
    return list(profiles.values())
