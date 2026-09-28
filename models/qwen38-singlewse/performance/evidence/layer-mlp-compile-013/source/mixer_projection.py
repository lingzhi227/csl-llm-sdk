"""Original mixed-projection addresses and one fixed adjacent reduction path.

Matrix-local time reuse stays inside one resident layer's mixer region. The
40- and48-part contractions share the same two neighbor colors; grouping is a
packet/compute condition, not a runtime router rewrite. Root output ports are
explicit interfaces for convolution, gates and the residual consumer.
"""
from spatial.layer_schedule import rank_xy,local_rows,tile_owner
from spatial.layer_routes import direction
from spatial.layer_mlp_network import broadcast


FIELDS=('base_words','iterations','first_row','row_stride','key_index','key_parts','bf16','rows')
SUFFIXES=('in_proj_qkv.weight','in_proj_z.weight','in_proj_a.weight','in_proj_b.weight','out_proj.weight')


def lower_mixer_projections(region,input_source):
    if region['role']!='mix' or len(region['matrices'])!=5:
        raise ValueError('Complete original GDN mixer matrix family required')
    matrices=region['matrices'];count=region['rect'][2]*region['rect'][3]
    expected=[([10240,5120],[2,128],'F8_E4M3'),([6144,5120],[2,128],'F8_E4M3'),
              ([48,5120],[1,128],'BF16'),([48,5120],[1,128],'BF16'),([5120,6144],[2,128],'F8_E4M3')]
    for m,suffix,(shape,tile,dtype) in zip(matrices,SUFFIXES,expected):
        if not m['tensor'].endswith('.linear_attn.'+suffix) or (m['shape'],m['tile_shape'],m['dtype'])!=(shape,tile,dtype):
            raise ValueError('Original projection identity or full dimensions changed')
        if m['k_blocks']!=shape[1]//128 or m['workers']!=m['groups']*m['k_blocks'] or 'group_partitions' in m:
            raise ValueError('Unexpected original mixer ownership')
    workers=[];ports=[]
    for rank in range(count):
        descriptors=[]
        for mi,m in enumerate(matrices):
            rows=m['tile_shape'][0];k=m['k_blocks'];group,key=divmod(rank,k)
            iterations=local_rows(m,rank);first=(group-m['group_rotation'])%m['groups']
            base=0
            if iterations:
                owner=tile_owner(region,mi,first*k+key)
                if owner['rank']!=rank or owner['byte_offset']%4:raise ValueError('Original bank base')
                base=owner['byte_offset']//4
                # Check every original tile address, including matrix-prefix
                # packing, rotation, tail rows and BF16 versus FP8 tile size.
                tile_words=64 if m['dtype']=='BF16' else 65
                for iteration in range(iterations):
                    original=tile_owner(region,mi,(first+iteration*m['groups'])*k+key)
                    if original['rank']!=rank or original['byte_offset']!=4*(base+iteration*tile_words):
                        raise ValueError('Native row loop changes original weight ownership')
                if key==0:
                    ports.append(dict(matrix=mi,pe=rank_xy(region,rank),first_row=first*rows,
                        row_stride=m['groups']*rows,iterations=iterations,rows=rows,complete_k=k,
                        output='BF16 after the complete FP32 K reduction',consumer=SUFFIXES[mi]))
            descriptors.append(dict(zip(FIELDS,[base,iterations,first*rows,m['groups']*rows,key,k,int(m['dtype']=='BF16'),rows])))
        workers.append(dict(pe=rank_xy(region,rank),rank=rank,descriptors=descriptors))
    routes=[]
    for rank in range(1,count):
        source=rank_xy(region,rank);destination=rank_xy(region,rank-1);color=10+rank%2
        routes.extend([dict(pe=source,color=color,rx='RAMP',tx=direction(source,destination)),
                       dict(pe=destination,color=color,rx=direction(destination,source),tx='RAMP')])
    inputs={}
    for color,mi in ((16,0),(17,4)):
        inputs[color]={tuple(w['pe']):w['descriptors'][mi]['key_index'] for w in workers if w['descriptors'][mi]['iterations']}
    input_routes=[r for color,recipients in inputs.items() for r in broadcast(input_source,recipients,color)]
    result=dict(region=region['id'],rect=region['rect'],fields=list(FIELDS),workers=workers,root_ports=ports,
        reduction_routes=routes,input_routes=input_routes,input_source=input_source,input_colors=[16,17],child_colors=[10,11],
        resources=dict(input40_queue=7,input48_queue=5,input48_queue_on_norm_sender=4,child_queue=6,send_queue=5,input_ut=7,child_ut=6,send_ut=5,
                       input_dsr=2,child_dsr=3,send_dsr=6,math_dsrs=[4,7]),
        full_mixer=False,executed=False,physical=False,
        missing='Initial input producer, root consumers, convolution/gates/state computation and complete stage scheduler')
    result['audit']=audit_mixer_projections(region,result)
    return result


def audit_mixer_projections(region,plan):
    matrix_stats=[];workers=plan['workers'];count=region['rect'][2]*region['rect'][3]
    if [w['rank'] for w in workers]!=list(range(count)) or [w['pe'] for w in workers]!=[rank_xy(region,r) for r in range(count)]:
        raise ValueError('Missing or duplicate mixer PE')
    routes={(*r['pe'],r['color']):r for r in plan['reduction_routes']}
    if len(routes)!=len(plan['reduction_routes']):raise ValueError('Neighbor reduction color alias')
    for mi,m in enumerate(region['matrices']):
        rows=[];tiles=0
        for w in workers:
            d=w['descriptors'][mi];rank=w['rank'];expected=local_rows(m,rank)
            if d['iterations']!=expected or d['key_index']!=rank%m['k_blocks'] or d['key_parts']!=m['k_blocks']:
                raise ValueError('Incomplete original mixed contraction')
            if d['bf16']!=int(m['dtype']=='BF16') or d['rows']!=m['tile_shape'][0]:raise ValueError('Original projection arithmetic type')
            for iteration in range(expected):
                row=d['first_row']+iteration*d['row_stride'];tile=(row//d['rows'])*m['k_blocks']+d['key_index']
                owner=tile_owner(region,mi,tile);stride=64 if d['bf16'] else 65
                if owner['pe']!=w['pe'] or owner['byte_offset']!=4*(d['base_words']+iteration*stride):
                    raise ValueError('Descriptor changes original tensor byte address')
                tiles+=1
                if d['key_index']==0:rows.extend(range(row,row+d['rows']))
            if expected and d['key_index']:
                p=rank_xy(region,rank-1);c=10+rank%2
                src=routes.get((*w['pe'],c));dst=routes.get((*p,c))
                if src!=dict(pe=w['pe'],color=c,rx='RAMP',tx=direction(w['pe'],p)) or dst!=dict(pe=p,color=c,rx=direction(p,w['pe']),tx='RAMP'):
                    raise ValueError('Disconnected complete-K neighbor reduction')
        if sorted(rows)!=list(range(m['shape'][0])) or tiles!=m['tiles']:
            raise ValueError('Missing or duplicated original projection rows/tiles')
        matrix_stats.append(dict(tensor=m['tensor'],shape=m['shape'],tiles=tiles,rows=len(rows)))
    expected_ports=[]
    for w in workers:
        for mi,d in enumerate(w['descriptors']):
            if d['iterations'] and d['key_index']==0:
                expected_ports.append((mi,w['pe'],d['first_row'],d['row_stride'],d['iterations'],d['rows'],d['key_parts']))
    if [(p['matrix'],p['pe'],p['first_row'],p['row_stride'],p['iterations'],p['rows'],p['complete_k']) for p in plan['root_ports']]!=expected_ports:
        raise ValueError('Root consumer port ownership')
    return dict(passed=True,original_matrices=matrix_stats,workers=count,fixed_neighbor_colors=[10,11],
                original_tiles_checked=sum(s['tiles'] for s in matrix_stats),full_mixer=False,physical=False)
