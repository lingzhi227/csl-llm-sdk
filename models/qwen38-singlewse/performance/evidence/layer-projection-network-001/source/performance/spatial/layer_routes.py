"""Actual native contraction routes inside the P23 resident MLP regions.

Balanced inorder trees choose a central physical-rank root. All express paths
stay inside a contraction's contiguous serpentine interval; depth/side colors
have disjoint intervals. This lowers reductions, not the entire layer fabric.
Root egress, sliced-input distribution and external credit ingress remain explicit
open interfaces. They cannot be replaced by an unconnected route count.
"""
from collections import Counter
from spatial.layer_schedule import local_rows,rank_xy,tile_owner


def reduction_tree(workers,blocks):
    if not 1<=workers<=blocks or blocks>65535:raise ValueError('Native contraction extent')
    nodes=[None]*workers
    def visit(first,stop,depth,parent,side):
        rank=(first+stop)//2;children=[]
        if first<rank:children.append(visit(first,rank,depth+1,rank,0))
        if rank+1<stop:children.append(visit(rank+1,stop,depth+1,rank,1))
        parts=len(range(rank,blocks,workers))
        node=dict(rank=rank,parent=parent,side=side,depth=depth,children=children,
                  parts=parts,subtree_k=parts+sum(nodes[c]['subtree_k'] for c in children))
        nodes[rank]=node;return rank
    root=visit(0,workers,0,None,None)
    assert nodes[root]['subtree_k']==blocks
    return root,nodes


def edge_color(node):
    if node['parent'] is None:raise ValueError('Root has a consumer, not a reduction parent')
    return 3+2*(node['depth']-1)+node['side']


def direction(source,dest):
    delta=tuple(b-a for a,b in zip(source,dest))
    return {(1,0):'EAST',(-1,0):'WEST',(0,1):'SOUTH',(0,-1):'NORTH'}[delta]


def contraction_groups(region):
    matrices=region['matrices'];matrix=matrices[0]
    if region['role'] not in ('gate_up','down') or 'group_partitions' not in matrix:
        raise ValueError('P23 resident MLP native groups required')
    if len(matrices)!=(2 if region['role']=='gate_up' else 1):raise ValueError('Matrix family')
    if any(m['group_partitions']!=matrix['group_partitions'] for m in matrices):raise ValueError('Paired owner mismatch')
    for index,g in enumerate(matrix['group_partitions']):
        root,nodes=reduction_tree(g['workers'],matrix['k_blocks'])
        yield dict(id=region['id']+'/K'+str(index),index=index,**g,
                   root=root,root_pe=rank_xy(region,g['rank_start']+root),nodes=nodes,
                   rows=matrix['tile_shape'][0],columns=matrix['tile_shape'][1],branches=len(matrices))


def worker(region,group,node):
    rank=group['rank_start']+node['rank'];rows=group['rows'];children=node['children']
    bases=[]
    for mi,matrix in enumerate(region['matrices']):
        tile=group['output_start']*matrix['k_blocks']+node['rank']
        owner=tile_owner(region,mi,tile)
        if owner['rank']!=rank or local_rows(matrix,rank)!=group['output_count']*node['parts']:
            raise ValueError('Runtime row loop differs from original weight addresses')
        bases.append(owner['byte_offset']//4)
    bank=next(c for c in region['banks']['classes'] if c['rank_start']<=rank<c['rank_end'])
    prefix=bank['page_prefix']+(rank-bank['rank_start'])*bank['auxiliary_pages']
    live=max(0,min(bank['auxiliary_pages'],region['auxiliary_pages']-prefix))
    payload=bank['fp8_slots']*260+bank['bf16_slots']*256+live*128
    if payload%4:raise ValueError('Unaligned original resident bank')
    return dict(pe=rank_xy(region,rank),rank=rank,group=group['id'],
        original_matrix_bytes=bank['fp8_slots']*260+bank['bf16_slots']*256,live_auxiliary_bytes=live*128,
        parameters=dict(rows=rows,columns=group['columns'],branches=group['branches'],
                        max_parts=group['max_k_parts'],bank_words=payload//4,root=node['parent'] is None,
                        children=len(children),subtree_k=node['subtree_k'],
                        left_k=group['nodes'][children[0]]['subtree_k'] if children else 0,
                        right_k=group['nodes'][children[1]]['subtree_k'] if len(children)>1 else 0,
                        left_color=edge_color(group['nodes'][children[0]]) if children else 3,
                        right_color=edge_color(group['nodes'][children[1]]) if len(children)>1 else 4,
                        out_color=19 if node['parent'] is None else edge_color(node)),
        arm=dict(parts=node['parts'],iterations=group['output_count'],base0=bases[0],base1=bases[1] if len(bases)>1 else 0,
                 first_output=group['output_start']*rows),
        native_input_slices=list(range(node['rank'],region['matrices'][0]['k_blocks'],group['workers'])))


def reduction_routes(region,group):
    for node in group['nodes']:
        if node['parent'] is None:continue
        source=node['rank'];parent=node['parent'];step=1 if parent>source else -1
        path=list(range(source,parent+step,step));coords=[rank_xy(region,group['rank_start']+n) for n in path]
        yield dict(source=coords[0],destination=coords[-1],color=edge_color(node),
                   words=3+group['rows']*group['branches'],subtree_k=node['subtree_k'],
                   routes=[dict(pe=xy,rx='RAMP' if i==0 else direction(xy,coords[i-1]),
                                tx='RAMP' if i==len(coords)-1 else direction(xy,coords[i+1])) for i,xy in enumerate(coords)])


def fused_deliveries(region):
    """Every actual full-K gate/up block goes to its original128-value group.

    The receiver credits each copied block immediately; waiting for the complete
    quantized group to credit a block would deadlock the producer's row loop.
    Physical paths and queue arbitration for these endpoints are not supplied.
    """
    if region['role']!='gate_up':raise ValueError('Gate/up projection required')
    actors={q:a for a in region['quantization_actors'] for q in a['groups']}
    for g in contraction_groups(region):
        for iteration in range(g['output_count']):
            first=(g['output_start']+iteration)*g['rows'];q=first//128
            if (first+g['rows']-1)//128!=q:raise ValueError('Native block crosses original quantization group')
            yield dict(producer=g['id'],source=g['root_pe'],destination=actors[q]['pe'],iteration=iteration,
                       first_output_row=first,quantization_group=q,first_pair=first%128//2,pairs=g['rows']//2,
                       packet_words=3+g['rows'],credit_words=2,
                       credit_after='All projected pairs copied/consumed into the active fusion group',physical_path=None)


def audit_mlp_network(plan):
    """Independently walk every tree, route and original K slice in all64 layers."""
    stats=Counter();colors=set();templates=set();maximum_hops=0
    for stage in plan['stages']:
        for region in stage['regions']:
            if region['role'] not in ('gate_up','down'):continue
            installed={};stats['regions']+=1
            for group in contraction_groups(region):
                stats['groups']+=1;templates.add((group['workers'],region['matrices'][0]['k_blocks']))
                nodes=group['nodes'];visited=set()
                def walk(rank):
                    if rank in visited:raise ValueError('Repeated/cyclic reduction owner')
                    visited.add(rank);n=nodes[rank]
                    slices=list(range(rank,region['matrices'][0]['k_blocks'],group['workers']))
                    for child in n['children']:
                        if nodes[child]['parent']!=rank:raise ValueError('Reduction parent mismatch')
                        slices.extend(walk(child))
                    if len(slices)!=n['subtree_k']:raise ValueError('Missing original K participants')
                    return slices
                original=walk(group['root'])
                if sorted(original)!=list(range(region['matrices'][0]['k_blocks'])) or len(visited)!=group['workers']:
                    raise ValueError('Not a complete original K reduction')
                for node in nodes:
                    w=worker(region,group,node);stats['workers']+=1
                    if any(base+group['output_count']*node['parts']*65>w['parameters']['bank_words'] for base in [w['arm']['base0']]+([w['arm']['base1']] if group['branches']==2 else [])):
                        raise ValueError('Native loop exceeds compiled bank')
                group_routes={}
                for flow in reduction_routes(region,group):
                    color=flow['color'];colors.add(color)
                    if not 3<=color<=17:raise ValueError('Reduction color collides with ingress/credit/egress or SDK')
                    records=flow['routes'];stats['edges']+=1;stats['word_hops_per_epoch']+=(len(records)-1)*flow['words']*group['output_count']
                    maximum_hops=max(maximum_hops,len(records)-1)
                    for route in records:
                        key=(*route['pe'],color);definition=(route['rx'],route['tx'])
                        if key in installed:raise ValueError('Two reduction paths share a PE/color')
                        installed[key]=definition;group_routes[key]=definition;stats['route_entries']+=1
                    # Follow the installed physical network, rather than just
                    # accepting the path endpoint labels used to construct it.
                    pe=tuple(flow['source']);incoming='RAMP';steps=0
                    delta={'EAST':(1,0),'WEST':(-1,0),'SOUTH':(0,1),'NORTH':(0,-1)}
                    opposite={'EAST':'WEST','WEST':'EAST','SOUTH':'NORTH','NORTH':'SOUTH'}
                    while True:
                        rx,tx=group_routes[(*pe,color)]
                        if rx!=incoming:raise ValueError('Disconnected reduction receive direction')
                        if tx=='RAMP':break
                        dx,dy=delta[tx];pe=(pe[0]+dx,pe[1]+dy);incoming=opposite[tx];steps+=1
                        if steps>group['workers']:raise ValueError('Physical routing cycle')
                    if list(pe)!=flow['destination']:raise ValueError('Wrong reduction consumer')
            if region['role']=='gate_up':
                pairs=Counter()
                for d in fused_deliveries(region):
                    stats['fused_projection_blocks']+=1
                    for pair in range(d['first_pair'],d['first_pair']+d['pairs']):pairs[d['quantization_group'],pair]+=1
                if len(pairs)!=136*64 or set(pairs.values())!={1}:raise ValueError('Fused original activation pair coverage')
    return dict(passed=True,**stats,tree_colors=sorted(colors),maximum_tree_edge_hops=maximum_hops,
                templates=sorted(templates),reduction_routes_connected=True,
                fusion_endpoint_coverage=True,complete_input_distribution=False,
                root_consumer_routes_connected=False,external_credit_routes_connected=False,
                all_layer_routes_admitted=False,numerical_qualified=False,physical=False)
