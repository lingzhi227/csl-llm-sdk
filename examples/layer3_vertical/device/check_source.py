"""Independent role, emitted-route, provenance and native-fanout checks."""
from collections import Counter,defaultdict
from pathlib import Path
import ast,hashlib,json,re

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'layer3-continuous-compile-002'
VERTICAL=HERE.parent/'layer0_vertical_top'

def check():
    plan,transport,host,prior=[json.loads((HERE/n).read_bytes()) for n in ('plan.json','transport-map.json','host-map.json','logical-plan.json')]
    original_transport=json.loads((OLD/'transport-map.json').read_bytes())
    assert (HERE/'logical-plan.json').read_bytes()==(OLD/'layer3-plan.json').read_bytes()
    w,h=plan['application'];assert (w,h)==(29,1160)
    cells={};matrix={};roles=defaultdict(set)
    def add(point,role):
        point=tuple(point);assert point not in cells and 0<=point[0]<w and 0<=point[1]<h
        cells[point]=role;roles[role].add(point);return point
    for name,role in [('origin',6),('input_norm',2),('post_norm',3),('observer',7)]:add(plan[name],role)
    for key,role in [('attention_heads',4),('mlp_owners',5),('head_observers',8)]:
        for p in plan[key]:add(p,role)
    mapping={tuple(e['legacy']):e for e in transport['endpoints']}
    assert set(mapping)==set(map(tuple,original_transport['endpoints']))
    semantic=('name','columns','phase','group','rows','last_columns','packet_kind','fanout')
    for i,(s,o) in enumerate(zip(plan['stripes'],prior['stripes'])):
        assert all(s[k]==o[k] for k in semantic) and s['logical_stripe']==i
        assert s['axis']=='y' and s['physical_step']==[0,1] and s['legacy_consumer']==o['consumer']
        assert s['consumer']==mapping[tuple(o['consumer'])]['endpoint']
        for ordinal in range(s['columns']):matrix[add((s['x'],s['y']+ordinal),1)]=(i,ordinal)
        if s['packet_kind'] in (2,3):assert s['router'] is None and (o['x'],o['y']) not in mapping
        else:assert mapping[o['x'],o['y']]==dict(router=s['router'],endpoint=[s['x'],s['y']],legacy=[o['x'],o['y']])
    for node in transport['nodes']:add((node['x'],node['y']),9)
    assert Counter(cells.values())=={1:30576,2:1,3:1,4:24,5:136,6:1,7:1,8:24,9:639}
    assert w*h-len(cells)==2237
    assert all(sum(abs(a-b) for a,b in zip(e['router'],e['endpoint']))==1 for e in mapping.values())
    assert {tuple(e['endpoint']) for e in mapping.values()}==({(s['x'],s['y']) for s in plan['stripes'] if s['packet_kind'] not in (2,3)}|roles[2]|roles[3]|roles[4]|roles[5]|roles[6])
    # Read the actual emitted CSL arrays and reconstruct every tile parameter.
    layout=(HERE/'layout.csl').read_text();arrays={}
    for name,n,body in re.findall(r'const (\w+):\[\d+\]u16=\[(\d+)\]u16\{([0-9,]+)\};',layout):
        arrays[name]=list(map(int,body.split(',')));assert len(arrays[name])==int(n)
    derived={}
    for x in range(w):
        for y in range(h):
            role=ordinal=kind=group=head=0;participants=54
            if x==0:
                if y in (1,2,3):role={1:6,2:7,3:2}[y]
                elif 84<=y<220:role=5
                elif 4<=y<676:
                    band,dy=divmod(y-4,220)
                    if dy<12:role=4 if dy%2==0 else 8;head=6*band+dy//2
            if x==28 and y==3:role=3
            for s in range(arrays['starts'][x],arrays['starts'][x+1]):
                if arrays['sy'][s]<=y<arrays['sy'][s]+arrays['lengths'][s]:
                    role=1;ordinal=y-arrays['sy'][s];participants=arrays['lengths'][s]
                    if ordinal==0:
                        kind=arrays['kinds'][s]
                        if kind in (2,3):group=arrays['groups'][s]
            index=next((i for i in range(arrays['router_start'][x+1]-arrays['router_start'][x]) if arrays['router_y'][arrays['router_start'][x]+i]==y),None)
            if index is not None or y==0:role=9
            assert role==cells.get((x,y),0)
            if role==1:
                s,o=matrix[x,y];expected=plan['stripes'][s]
                assert ordinal==o and participants==expected['columns']
                assert kind==(expected['packet_kind'] if o==0 else 0)
                assert group==(expected['group'] if o==0 and kind in (2,3) else 0)
            if role in (4,8):assert [x,y]==plan['attention_heads' if role==4 else 'head_observers'][head]
            derived[x,y]=role
    assert '.memcpy_params=memcpy.get_params(x)' in layout and 'memcpy.get_params(y)' not in layout

    # Expand literal/segment color declarations from CSL, not transport metadata.
    routes={};owner={};steps=dict(NORTH=(0,-1),SOUTH=(0,1),EAST=(1,0),WEST=(-1,0));opposite=dict(NORTH='SOUTH',SOUTH='NORTH',EAST='WEST',WEST='EAST')
    def route(x,y,color,rx,tx,why):
        key=(x,y,color);assert key not in routes,(key,why,owner.get(key))
        routes[key]=(rx,tuple(tx));owner[key]=why
    def coord(expression,i):
        match=re.fullmatch(r'(\d+)([+-])i',expression)
        if match:return int(match[1])+(i if match[2]=='+' else -i)
        assert expression.isdigit();return int(expression)
    pattern=r' for\(@range\(u16,(\d+)\)\)\|i\|\{@set_color_config\(([^,]+),([^,]+),@get_color\((\d+)\),\.\{\.routes=\.\{\.rx=if\(i==0\) \.\{RAMP\} else \.\{(\w+)\},\.tx=if\(i==(\d+)\) \.\{RAMP\} else \.\{(\w+)\}\}\}\);\}'
    emitted_segments=re.findall(pattern,layout);assert len(emitted_segments)==len(transport['segments'])
    for count,x,y,color,rx,last,tx in emitted_segments:
        assert int(last)+1==int(count)
        for i in range(int(count)):route(coord(x,i),coord(y,i),int(color),'RAMP' if i==0 else rx,['RAMP'] if i==int(last) else [tx],'emitted_segment')
    literal=r' @set_color_config\((\d+),(\d+),@get_color\((\d+)\),\.\{\.routes=\.\{\.rx=\.\{(\w+)\},\.tx=\.\{([A-Z,]*)\}\}\}\);'
    emitted_literals=re.findall(literal,layout)
    assert len(emitted_literals)==w+len(transport['native_routes'])
    for x,y,color,rx,tx in emitted_literals:route(int(x),int(y),int(color),rx,tx.split(',') if tx else [],'emitted_literal')
    stop_pairs=re.findall(r'if\(matrix\)\{@set_color_config\((\d+),y,@get_color\(4\),\.\{\.routes=\.\{\.rx=\.\{NORTH\},\.tx=if\(y==(\d+)\)',layout)
    assert len(stop_pairs)==w-2
    for x,stop in map(lambda v:tuple(map(int,v)),stop_pairs):
        assert stop==max(y for xx,y in roles[1] if xx==x)
        for y in range(2,stop+1):route(x,y,4,'NORTH',(['RAMP'] if (x,y) in roles[1] else [])+(['SOUTH'] if y<stop else []),'emitted_frame_column')
    for s in plan['stripes']:
        for ordinal in range(s['columns']):
            x,y=s['x'],s['y']+ordinal
            if ordinal<s['columns']-1:route(x,y,ordinal%2,'SOUTH',['RAMP'],'matrix_import')
            if ordinal:route(x,y,(ordinal+1)%2,'RAMP',['NORTH'],'matrix_import')
            route(x,y,2,'RAMP' if ordinal==0 else 'NORTH',['RAMP'] if ordinal==s['columns']-1 else ['SOUTH'] if ordinal==0 else ['RAMP','SOUTH'],'matrix_import')
    links=0
    for (x,y,color),(rx,txs) in routes.items():
        for tx in txs:
            if tx=='RAMP':continue
            dx,dy=steps[tx];assert routes[x+dx,y+dy,color][0]==opposite[tx];links+=1
    assert len(routes)==transport['summary']['route_configurations'] and links==transport['summary']['physical_links']
    # Trace all16 actual native trees through emitted colors to all96 recipients.
    stream_cells=defaultdict(set);native_hops=0
    for stream in transport['native_streams']:
        s=plan['stripes'][stream['logical_stripe']];assert [s['x'],s['y']]==stream['source'] and s['packet_kind']==stream['kind'] and s['group']==stream['group']
        group=stream['group']//2;assert stream['head_ids']==list(range(6*group,6*group+6))
        pending=[tuple(stream['source'])];seen=set();deliveries=set();color=stream['color']
        assert routes[*stream['source'],color][0]=='RAMP'
        while pending:
            point=pending.pop();assert point not in seen;seen.add(point)
            for tx in routes[*point,color][1]:
                if tx=='RAMP':deliveries.add(point)
                else:
                    dx,dy=steps[tx];pending.append((point[0]+dx,point[1]+dy));native_hops+=1
        assert deliveries=={tuple(plan['attention_heads'][h]) for h in stream['head_ids']}
        assert not seen&stream_cells[color];stream_cells[color]|=seen
    assert len(stream_cells)==4
    # Router forwarding selector independently reproduces each original message.
    nodes={(n['x'],n['y']):n for n in transport['nodes']};actual_counts=Counter();hop_sum=0;longest=0
    parent={tuple(e['child']):tuple(e['parent']) for e in transport['edges']}
    children=defaultdict(list)
    for a,b in parent.items():children[b].append(a)
    ports={}
    for point,node in nodes.items():
        if node['spine']:
            down=next((p for p in children[point] if p[0]==point[0]),None)
            ports[point]={0:down,1:(point[0]-1,0) if point[0] else None,2:(point[0]+1,0) if point[0]<w-1 else None}
        else:
            down=next(iter(children[point]),None);ports[point]={0:None,1:down,2:parent[point]}
    for flow,oldflow in zip(transport['traffic'],original_transport['traffic']):
        a=tuple(mapping[tuple(oldflow['source'])]['router']);b=tuple(mapping[tuple(oldflow['destination'])]['router'])
        assert list(a)==flow['source'] and list(b)==flow['destination'] and flow['kind']==oldflow['kind'] and flow['packets']==oldflow['packets']
        point=a;visited=set();distance=2
        while point!=b:
            assert point not in visited;visited.add(point)
            if nodes[point]['spine']:port=0 if b[0]==point[0] else 1 if b[0]<point[0] else 2
            else:port=1 if b[0]==point[0] and b[1]>point[1] else 2
            other=ports[point][port];assert other is not None and nodes[point]['mask']&(1<<port)
            actual_counts[point,other]+=flow['packets'];distance+=sum(abs(x-y) for x,y in zip(point,other));point=other
        hop_sum+=distance*flow['packets'];longest=max(longest,distance)
    assert actual_counts==Counter({(tuple(e['source']),tuple(e['destination'])):e['packets'] for e in transport['expected_directed_edge_packets']})
    assert hop_sum==1394298 and longest==1792
    # Independently map each original source batch row to all physical PE words.
    old_sources={}
    for batch_id,batch in enumerate(prior['weight_batches']):
        for row,group in enumerate(batch['row_groups']):old_sources[batch['name'],group]=(f'weights-{batch_id:03}.npy',row,batch['width'])
    uploaded={}
    for item in host['matrix_uploads']:
        actual={}
        for seg in item['segments']:
            s=plan['stripes'][seg['logical_stripe']];file,row,width=old_sources[s['name'],s['group']]
            assert seg['original_file']==file and seg['original_row']==row and width==s['columns']
            for offset in range(seg['tile_count']):
                xy=(item['x']+seg['destination_x'],item['y']+seg['destination_y']+offset)
                value=(seg['logical_stripe'],seg['ordinal_begin']+offset);assert value==matrix[xy] and xy not in actual;actual[xy]=value
        expected={(x,y) for y in range(item['y'],item['y']+item['height']) for x in range(item['x'],item['x']+item['width'])}
        assert set(actual)==expected and not set(actual)&set(uploaded);uploaded.update(actual)
        assert item['host_bytes']==len(actual)*6144*4<=16<<20
    assert uploaded==matrix and sum(i['host_bytes'] for i in host['matrix_uploads'])==751435776
    assert len(host['persistent_exports'])==24
    for head,item in enumerate(host['persistent_exports']):
        assert [item['x'],item['y']]==plan['attention_heads'][head] and item['head']==head and item['count']==4096
    assert sum(i['native_bytes'] for i in host['persistent_exports'])==196608
    # Only documented physical substitutions can differ in the accepted wrapper.
    pe=(HERE/'layer3_pe.csl').read_text().replace('const support_coordinates=@import_module("support_coordinates.csl",.{});\n','')
    changes=[('rows.transmit(support_coordinates.x(lane.descriptor[2]+fanout_index),support_coordinates.y(lane.descriptor[2]+fanout_index),root_values);','rows.transmit(lane.descriptor[2]+fanout_index,lane.descriptor[3],root_values);'),
        ('packets.send(support_coordinates.x(request_x),support_coordinates.y(request_x),&tx,8);','packets.send(request_x,request_y,&tx,8);'),
        ('rows.transmit(support_coordinates.x(2),support_coordinates.y(2),&retained_chunk);','rows.transmit(2,0,&retained_chunk);')]
    for a,b in changes:assert pe.count(a)==1;pe=pe.replace(a,b)
    assert pe==(OLD/'layer3_pe.csl').read_text()
    assert (HERE/'matrix_lane.csl').read_text().replace('.negative=NORTH,.positive=SOUTH','.negative=WEST,.positive=EAST')==(OLD/'matrix_lane.csl').read_text()
    for n in ('packets.csl','transport_header.csl'):assert (HERE/n).read_bytes()==(VERTICAL/n).read_bytes()
    changed={'layout.csl','matrix_lane.csl','tree_router.csl','packets.csl','transport_header.csl','native_colors.csl','credit_coordinates.csl','credit_head_coordinates.csl','layer3_pe.csl'}
    exact=[]
    for p in OLD.glob('*.csl'):
        if p.name not in changed:assert p.read_bytes()==(HERE/p.name).read_bytes();exact.append(p.name)
    for p in HERE.glob('*.csl'):
        for name in re.findall(r'"([^"\n]+\.csl)"',p.read_text()):assert (HERE/name).is_file()
    for p in HERE.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
    result=dict(status='generated_Layer3_vertical_source_checks_passed_not_compiled',application=[w,h],
        matrix_tiles=30576,logical_stripes=464,exact_original_weight_word_slots=30576*6144,
        source_weight_batches=95,weight_ROIs=len(host['matrix_uploads']),all_ROIs_role_exact=True,
        all_emitted_color_entries=len(routes),all_physical_links=links,native_streams=16,native_deliveries=96,
        native_hops_per_word_total=native_hops,same_color_native_regions_disjoint=True,
        packet_paths=1223,packet_hops=hop_sum,longest_packet_path=longest,
        full_KV_native_bytes=196608,exact_prior_CSL=sorted(exact),new_header_and_packet_source='Byte-exact accepted vertical Layer0',
        wrapper_delta='One physical-coordinate import and three destination translations; identity/state/math unchanged',
        SDK_invocations=0,compiled=False,adjacent_handoff_implemented=False,control_restore_implemented=False,full_model=False)
    (HERE/'SOURCE-CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':check()
