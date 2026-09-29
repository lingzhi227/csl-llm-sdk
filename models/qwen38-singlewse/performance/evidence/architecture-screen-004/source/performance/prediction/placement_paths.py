"""Complete owner-to-owner paths for a declared cross-layer handoff subgraph.

Uses native down reduction roots and the next mix region's allocated slot pages.
The path includes producer-to-boundary and boundary-to-consumer, not just its
middle hop. These geometric routes have no color/queue admission and are only
one part of the dependent full-model path. Counts aggregate every color onto
the same directed physical link. No hop count is converted to measured time.
"""
from collections import Counter
from spatial.layer_schedule import auxiliary_owner
from spatial.layer_routes import contraction_groups
from spatial.pipeline_lowering import boundary_ports


def manhattan(a,b):
    p=list(a);result=[tuple(p)]
    for axis in (0,1):
        while p[axis]!=b[axis]:
            p[axis]+=1 if p[axis]<b[axis]else -1
            result.append(tuple(p))
    return result


def owner_flows(plan):
    """Exact original 5120 BF16 rows for each of the 63 layer-to-layer edges.

    The candidate epilogue emits contiguous rows from complete-K down roots.
    Destinations own 64-value BF16 slot pages; a 128-value quantization chunk
    can span two owners. Five header words match the pipeline wire contract.
    Residual availability at these roots is an explicit unlowered prerequisite.
    """
    for source,destination in zip(plan['stages'][1:-2],plan['stages'][2:-1]):
        down=next(r for r in source['regions']if r['role']=='down')
        mix=next(r for r in destination['regions']if r['role']=='mix')
        first=mix['nodes'][0];value=first['inputs'][0]
        slot=next(a for a in mix['auxiliary']if a['kind']=='slot_value'and a['id']==value)
        if slot['bytes_per_slot']!=5120*2:
            raise ValueError('Next input slot differs from the original 5120 BF16 vector')
        seen=set()
        for group in contraction_groups(down):
            row=group['output_start']*group['rows']
            end=row+group['output_count']*group['rows']
            while row<end:
                count=min(end-row,64-row%64)
                if seen.intersection(range(row,row+count)):
                    raise ValueError('Duplicate original output rows')
                seen.update(range(row,row+count))
                page=slot['page_start']+row//64
                owner=auxiliary_owner(mix,page)
                yield dict(source_stage=source['id'],destination_stage=destination['id'],
                    source_rect=source['rect'],destination_rect=destination['rect'],
                    source=group['root_pe'],destination=owner['pe'],
                    destination_byte_offset=owner['byte_offset']+(row%64)*2,
                    first=row,count=count,chunk=row//128,words=5+(count+1)//2)
                row+=count
        if seen!=set(range(5120)):
            raise ValueError('Incomplete native output coverage')


def route_handoffs(plan, policy):
    if policy not in ('fixed_lane','owner_aligned'):
        raise ValueError('Declared boundary policy required')
    loads=Counter();hop_words=0;packets=0;words=0;max_hops=0;edges={}
    for f in owner_flows(plan):
        ports=boundary_ports(f['source_rect'],f['destination_rect'])
        def path(p):
            return manhattan(f['source'],p['source'])+manhattan(p['destination'],f['destination'])
        if policy=='fixed_lane':
            selected=ports[f['chunk']%len(ports)];route=path(selected)
        else:
            options=[path(p)for p in ports]
            # Greedy tie-break balances physical-link words, not fabric colors.
            route=min(options,key=lambda r:(len(r),max((loads[a+b]+f['words']for a,b in zip(r,r[1:])),default=0),r))
        directed=[a+b for a,b in zip(route,route[1:])]
        if any(abs(a-c)+abs(b-d)!=1 for a,b,c,d in directed):
            raise ValueError('Non-neighbor route')
        for link in directed:loads[link]+=f['words']
        n=len(directed);hop_words+=n*f['words'];packets+=1;words+=f['words'];max_hops=max(max_hops,n)
        edge=edges.setdefault(f['source_stage'],dict(maximum_path_hops=0,packets=0,payload_rows=0))
        edge['maximum_path_hops']=max(edge['maximum_path_hops'],n)
        edge['packets']+=1;edge['payload_rows']+=f['count']
    if len(edges)!=63 or any(e['payload_rows']!=5120 for e in edges.values()):
        raise ValueError('Expected every complete layer-to-layer vector')
    return dict(policy=policy,layer_edges=63,original_bf16_rows=63*5120,packets=packets,words=words,
                weighted_hop_words=hop_words,mean_word_distance=hop_words/words,
                maximum_path_hops=max_hops,used_directed_links=len(loads),
                maximum_directed_link_words=max(loads.values()),
                sum_of_per_edge_maximum_hops=sum(e['maximum_path_hops']for e in edges.values()),
                edge_geometry=edges,
                top_links=[dict(source=list(k[:2]),destination=list(k[2:]),words=v)for k,v in loads.most_common(10)],
                scope='63 exact native down-root to next allocated input-page handoffs, including both interior paths; '
                      'residual colocation, RMS allreduce and subsequent operand fanout remain prerequisites',
                complete_model_network=False,color_queue_admitted=False,measured_latency=False)
