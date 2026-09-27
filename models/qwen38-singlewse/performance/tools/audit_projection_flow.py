"""Independent route traversal, traffic, partition and completion audit."""
from collections import Counter
from graphlib import TopologicalSorter


def audit(plan):
    width,height=plan['application'];spec=plan['spec'];workers=plan['workers'];streams=plan['streams']
    rows,columns=spec['tile_rows'],spec['tile_columns'];groups=spec['output_rows']//rows;kblocks=spec['input_columns']//columns
    xy={tuple(w['xy']):w for w in workers};assert len(xy)==len(workers)==groups*kblocks
    assert {(w['group'],w['k']) for w in workers}=={(g,k) for g in range(groups) for k in range(kblocks)}
    for w in workers:
        assert w['row_start']==w['group']*rows and w['column_start']==w['k']*columns
    routes={(*r['xy'],r['color']):r for r in plan['routes']};assert len(routes)==len(plan['routes'])
    directions=dict(NORTH=(0,-1),SOUTH=(0,1),EAST=(1,0),WEST=(-1,0));opposite=dict(NORTH='SOUTH',SOUTH='NORTH',EAST='WEST',WEST='EAST')
    covered=set();loads=Counter();counts=Counter();max_input_hops=0
    for stream in streams:
        origin=tuple(stream['path'][0]);start=(*origin,stream['color']);assert routes[start]['rx']=='RAMP'
        pending=[(start,0,frozenset())];delivered=set();hops=0
        while pending:
            here,distance,ancestors=pending.pop();assert here not in ancestors and here not in covered
            covered.add(here);x,y,color=here;assert 0<=x<width and 0<=y<height and 2<=color<20
            r=routes[here];assert r['stream']==stream['id'];assert r['tx'] and len(r['tx'])==len(set(r['tx']))
            assert r['teardown']==(stream['kind']=='input') and not r['filtered']
            for port in r['tx']:
                if port=='RAMP':
                    assert (x,y) not in delivered;delivered.add((x,y))
                    if stream['kind']=='input':max_input_hops=max(max_input_hops,distance)
                    continue
                dx,dy=directions[port];there=(x+dx,y+dy,color);assert routes[there]['rx']==opposite[port]
                loads[x,y,x+dx,y+dy]+=stream['words']+stream['control_words'];hops+=1
                pending.append((there,distance+1,ancestors|{here}))
        assert delivered=={tuple(p) for p in stream['consumers']}
        counts[stream['kind']]+=hops*(stream['words']+stream['control_words'])
        if stream['kind']=='input':
            producer=next(p for p in plan['input_owners'] if tuple(p['xy'])==origin)
            k=producer['k'];assert stream['words']==columns//2+1 and stream['control_words']==1
            assert delivered=={tuple(w['xy']) for w in workers if w['k']==k}
    assert covered==set(routes)
    actual={(tuple(v['source'])+tuple(v['destination'])):v['words'] for v in plan['cost']['link_loads']}
    assert actual==dict(loads) and dict(counts)==plan['cost']['word_hops_by_kind']
    assert max(loads.values())==plan['cost']['max_link_words_including_teardown']
    assert max_input_hops==plan['cost']['max_input_hops']
    byid={v['id']:v for v in streams};assert len(byid)==len(streams)
    byrank={w['rank']:w for w in workers};assert set(byrank)==set(range(len(workers)))
    kick=byid['kick'];assert kick['path'][0]==[0,0] or kick['path'][0]==(0,0)
    assert {tuple(v) for v in kick['consumers']}=={tuple(v['xy']) for v in plan['input_owners']}
    assert kick['words']==1 and kick['control_words']==0
    delivery=byid['delivery'];assert tuple(delivery['path'][0])==tuple(byrank[0]['xy']) and {tuple(v) for v in delivery['consumers']}=={(0,0)}
    assert delivery['words']==spec['output_rows'] and delivery['control_words']==0
    events=plan['completion_dag'];assert set(TopologicalSorter(events).static_order())==set(events)
    # Reconstruct every native completion interval, without importing tree().
    for g in range(groups):
        pending=[(0,kblocks)]
        while pending:
            rank,size=pending.pop();left=size//2;right=size-1-left;children=[]
            if left:children.append(rank+1);pending.append((rank+1,left))
            if right:children.append(rank+1+left);pending.append((rank+1+left,right))
            assert events[f'sum:{g*kblocks+rank}']==[f'input:{rank}']+[f'sum:{g*kblocks+c}' for c in children]
            for child in children:
                stream=byid[f'native:{g*kblocks+child}']
                assert tuple(stream['path'][0])==tuple(byrank[g*kblocks+child]['xy'])
                assert {tuple(v) for v in stream['consumers']}=={tuple(byrank[g*kblocks+rank]['xy'])}
                assert stream['words']==rows+1 and stream['control_words']==0
    assert all(events[f'input:{k}']==['kick'] for k in range(kblocks))
    assert events['kick']==['inputs-ready'] and events['inputs-ready']==[] and events['delivered']==['gather:0']
    for g in range(groups):
        children=[];step=1
        while g%(2*step)==0 and g+step<groups:children.append(g+step);step*=2
        assert events[f'gather:{g}']==[f'sum:{g*kblocks}']+[f'gather:{c}' for c in children]
        for child in children:
            stream=byid[f'gather:{child}'];assert tuple(stream['path'][0])==tuple(byrank[child*kblocks]['xy'])
            assert {tuple(v) for v in stream['consumers']}=={tuple(byrank[g*kblocks]['xy'])}
            assert stream['words']==rows*min(child&-child,groups-child) and stream['control_words']==0
    return dict(passed=True,physical=False,workers=len(workers),input_owners=len(plan['input_owners']),streams=len(streams),routes=len(routes),directed_links=len(loads),
                max_input_hops=max_input_hops,max_link_words_including_teardown=max(loads.values()),word_hops_by_kind=dict(counts),
                exact_route_coverage=True,partition_and_completion_dependencies_checked=True,compiled_sram_qualified=False,
                scope='Static independently traversed actual routes/traffic/partition/completion edges. Does not prove runtime liveness, numerical results or performance.')
