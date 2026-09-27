"""Independent reverse-owner and graph-traversal oracle for a complete matrix slice."""
import argparse,hashlib,json
from pathlib import Path

def audit(base,overlay,plan):
    matrix=next(m for m in overlay['matrices'] if m['tensor']=='model.language_model.layers.0.linear_attn.in_proj_a.weight')
    assert plan['matrix']==matrix and matrix['shape']==[48,5120]
    assert plan['revision']==base['revision']==overlay['revision'] and plan['model']==base['model']==overlay['model']
    workers=plan['workers'];width,height=plan['application'];ox,oy=plan['global_origin']
    indexed={tuple(w['xy']):w for w in workers};assert len(indexed)==len(workers)==960
    observed=set();row_index={y:i for i,y in enumerate(base['geometry']['bank_rows'])}
    # Invert every physical coordinate rather than using candidate rank -> xy.
    for y in range(height):
        for x in range(width):
            gx,gy=x+ox,y+oy
            if gy not in row_index:continue
            row=row_index[gy];bank=row*750+(gx if row%2==0 else 749-gx)
            matches=[s for s in base['bf16_eligible_segments'] if s['bank_start']<=bank<s['bank_start']+s['count']]
            if not matches:continue
            assert len(matches)==1;s=matches[0];eligible=s['compact_start']+bank-s['bank_start']
            tile=(eligible-overlay['classes']['bf16']['phase']-matrix['stream_start'])%824000
            if tile>=960:continue
            w=indexed[x,y];observed.add(tile)
            assert w['rank']==tile and w['group']==tile//40 and w['k']==tile%40
            assert w['global_xy']==[gx,gy] and w['bank_rank']==bank and w['eligible_rank']==eligible
            assert w['target_slot']==(matrix['stream_start']+tile)//824000==6
    assert observed==set(range(960))
    directions=dict(NORTH=(0,-1),SOUTH=(0,1),EAST=(1,0),WEST=(-1,0));opposite=dict(NORTH='SOUTH',SOUTH='NORTH',EAST='WEST',WEST='EAST')
    routes={(*r['xy'],r['color']):r for r in plan['routes']};assert len(routes)==len(plan['routes'])
    for (x,y,c),r in routes.items():
        assert 0<=x<width and 0<=y<height and r['rx'] in ['RAMP',*directions]
        assert len(set(r['tx']))==len(r['tx']) and r['tx']
        for direction in r['tx']:
            if direction=='RAMP':continue
            dx,dy=directions[direction];neighbor=routes[x+dx,y+dy,c]
            assert neighbor['rx']==opposite[direction]
    # Actual route traversal must terminate, cover every entry and deliver only
    # to the declared recipients. It does not assume a serpentine path.
    paths={};covered=set();input_consumers=set();input_forwarders=set();input_max_hops=0
    for key,route in routes.items():
        if route['rx']!='RAMP':continue
        color=key[2];pending=[(key,0,frozenset())];ends=[]
        while pending:
            here,hops,ancestors=pending.pop();assert here not in ancestors;covered.add(here)
            x,y,c=here;r=routes[here]
            if c==2:input_forwarders.add((x,y))
            for direction in r['tx']:
                if direction=='RAMP':
                    ends.append(((x,y),hops))
                    if c==2:input_consumers.add((x,y));input_max_hops=max(input_max_hops,hops)
                else:
                    dx,dy=directions[direction];pending.append(((x+dx,y+dy,c),hops+1,ancestors|{here}))
        if color!=2:
            assert len(ends)==1;paths[key]=(ends[0][0],ends[0][1])
    assert covered==set(routes) and input_consumers==set(indexed)
    assert {tuple(r['xy']) for r in routes.values() if r['filtered']}==input_consumers
    assert input_forwarders=={tuple(xy) for xy in plan['input_path']}
    expected={};groups=24
    for group in range(groups):
        pending=[(0,40,0)]
        while pending:
            parent,size,depth=pending.pop();left=size//2;right=size-1-left
            for side,child,count in [(0,parent+1,left),(1,parent+1+left,right)]:
                if count:
                    source=workers[group*40+child]['xy'];dest=workers[group*40+parent]['xy'];expected[(*source,3+2*depth+side)]=(tuple(dest),3)
                    pending.append((child,count,depth+1))
    native_edges=len(expected)
    stage=0
    while 2**stage<groups:
        step=2**stage
        for parent in range(0,groups,2*step):
            child=parent+step
            if child>=groups:continue
            source=workers[child*40]['xy'];dest=workers[parent*40]['xy'];expected[(*source,12+stage)]=(tuple(dest),2*min(step,groups-child))
        stage+=1
    expected[(*workers[0]['xy'],18)]=((0,0),48)
    assert set(expected)==set(paths)
    for key,(destination,words) in expected.items():assert paths[key][0]==destination
    return dict(schema='wse-complete-matrix-routing-audit-v1',passed=True,physical=False,weights_read=False,matrix_shape=[48,5120],all_original_tiles=960,contraction_edges=native_edges,gather_edges=len(expected)-native_edges-1,
                route_entries=len(routes),input_recipients=len(input_consumers),input_forwarders=len(input_forwarders),input_words=2600,source_to_input_farthest_hops=input_max_hops,
                contraction_and_gather_word_hops=sum(words*paths[key][1] for key,(destination,words) in expected.items()),
                all_routes_reachable=True,all_intermediate_router_ports_checked=True,all_original_addresses_and_output_rows_checked=True,
                compiled_sram_admitted=False,full_model_executable=False,scope='Complete original matrix owner and static input/contraction/ordered-output routing only. Abstract graph traversal is not executable queue/thread/numerical qualification.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(*[json.loads(path.read_text()) for path in [a.base,a.overlay,a.plan]])
    result['sha256']={n:hashlib.sha256(path.read_bytes()).hexdigest() for n,path in [('base',a.base),('overlay',a.overlay),('plan',a.plan)]}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
