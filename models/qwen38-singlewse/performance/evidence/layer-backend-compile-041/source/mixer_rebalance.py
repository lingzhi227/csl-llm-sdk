"""Joint original QKV/Z/output row ownership under measured cohost limits.

All K128 blocks of a row remain in the same40/48-PE contraction. The overlapping
40- and48-PE groups form independent240-PE components. An integer dynamic
program trades output rows within each component, then joins component sums;
no model arrays, floating-point solver, or relaxed integer solution is involved.
"""
from collections import defaultdict
from copy import deepcopy
from spatial.layer_schedule import local_rows, rank_xy, tile_owner
from spatial.mixer_projection import FIELDS


def balanced_counts(total, capacities, minimum=1):
    if any(c<minimum for c in capacities) or not minimum*len(capacities)<=total<=sum(capacities):
        raise ValueError('Complete integer row capacity is insufficient')
    lo=minimum;hi=max(capacities)
    while lo<hi:
        mid=(lo+hi+1)//2
        if sum(min(c,mid) for c in capacities)<=total:lo=mid
        else:hi=mid-1
    result=[min(c,lo) for c in capacities];remaining=total-sum(result)
    for i,c in enumerate(capacities):
        if remaining and result[i]<c:result[i]+=1;remaining-=1
    assert remaining==0
    return result


def row_capacities(region, bank_limits, compact_scales=False):
    """x[g]=QKV+Z pairs; y[h]=output pairs. Each edge constrains x+y."""
    if region['rect']!=[63,67,45,79]:raise ValueError('Pinned layer00 mixer required')
    matrices=region['matrices'];edges={};y_limits={};x_limits={};out_groups=matrices[4]['groups']
    if [m['k_blocks'] for m in matrices]!=[40,40,40,40,48]:raise ValueError('Original K extents changed')
    for rank in range(3555):
        pe=tuple(rank_xy(region,rank))
        if pe not in bank_limits:continue  # The original final gateway has no matrix bank.
        bf16=256*sum(local_rows(matrices[i],rank) for i in (2,3))
        # With contiguous ownership, at most16 original block scales cover
        # the bounded local QKV/Z/output intervals. Exact tables are smaller;
        # the final compact-address audit checks this conservative reservation.
        cap=(bank_limits[pe]-bf16-(64 if compact_scales else 0))//(256 if compact_scales else 260)
        g=rank//40;h=rank//48
        if g<88 and h<out_groups:edges[g,h]=min(edges.get((g,h),cap),cap)
        elif g<88:x_limits[g]=min(x_limits.get(g,cap),cap)
        elif h<out_groups:y_limits[h]=min(y_limits.get(h,cap),cap)
        elif cap<0:raise ValueError('Negative empty-PE capacity')
    if {g for g,h in edges}|set(x_limits)!=set(range(88)) or {h for g,h in edges}|set(y_limits)!=set(range(out_groups)):
        raise ValueError('Incomplete original group capacity')
    return edges,y_limits,x_limits


def solve_counts(region, bank_limits, *, radius=4, compact_scales=False):
    if type(radius) is not int or not 1<=radius<=16:raise ValueError('Bounded integer search radius')
    edges,y_limits,x_limits=row_capacities(region,bank_limits,compact_scales);out_groups=region['matrices'][4]['groups']
    adjacent=defaultdict(list)
    for (g,h),cap in edges.items():adjacent[g].append((h,cap))
    for neighbors in adjacent.values():
        neighbors.sort()
        if len(neighbors) not in (1,2) or (len(neighbors)==2 and neighbors[1][0]!=neighbors[0][0]+1):
            raise ValueError('Original contraction overlap is not a chain')
    endings=defaultdict(list)
    for g,neighbors in adjacent.items():endings[neighbors[-1][0]].append((g,neighbors))
    previous=[local_rows(region['matrices'][4],h*48) for h in range(out_groups)]
    choices={h:range(max(1,previous[h]-radius),min(previous[h]+radius,y_limits.get(h,65535))+1) for h in range(out_groups)}
    if any(not c for c in choices.values()):raise ValueError('Output cohost has no row capacity in the bounded search')
    components=[]
    # Each record is(maximum x sum, negative movement cost, y values). Comparing
    # the first two fields maximizes capacity, then preserves original work size.
    def better(a,b):return b is None or a[:2]>b[:2]
    for start in range(0,out_groups,5):
        stop=min(out_groups,start+5);dp={(0,None):(0,0,())}
        for h in range(start,stop):
            after={}
            for (total,last),(score,penalty,values) in dp.items():
                for y in choices[h]:
                    contribution=0;valid=True
                    for g,neighbors in endings[h]:
                        candidates=[cap-(y if hh==h else last) for hh,cap in neighbors]
                        capacity=min(candidates+[x_limits.get(g,65535)])
                        if capacity<2:valid=False;break
                        contribution+=capacity
                    if not valid:continue
                    item=(score+contribution,penalty-abs(y-previous[h]),values+(y,));key=(total+y,y)
                    if better(item,after.get(key)):after[key]=item
            dp=after
        summary={}
        for (total,_),item in dp.items():
            if better(item,summary.get(total)):summary[total]=item
        if not summary:raise ValueError('No integer layout in a contraction component')
        components.append(summary)
    isolated=sum(cap for g,cap in x_limits.items() if g not in adjacent)
    joined={0:(isolated,0,())}
    for i,component in enumerate(components):
        after={};future=components[i+1:]
        minimum=sum(min(c) for c in future);maximum=sum(max(c) for c in future)
        for total,(score,penalty,values) in joined.items():
            for extra,(more,cost,ys) in component.items():
                key=total+extra
                if key+minimum>2560 or key+maximum<2560:continue
                item=(score+more,penalty+cost,values+ys)
                if better(item,after.get(key)):after[key]=item
        joined=after
    if 2560 not in joined:raise ValueError('Complete output matrix cannot fit the bounded integer layout')
    maximum,penalty,y=joined[2560]
    caps=[min([cap-y[h] for h,cap in adjacent[g]]+[x_limits.get(g,65535)]) for g in range(88)]
    assert maximum==sum(caps)
    if maximum<8192:
        raise ValueError('Complete QKV/Z needs8192 pairs but bounded layout admits'+str(maximum))
    total=balanced_counts(8192,caps,minimum=2)
    q=[max(1,min(t-1,t*5//8)) for t in total]
    remaining=5120-sum(q)
    for i in sorted(range(88),key=lambda i:(-(total[i]*5%8),i)):
        if remaining and q[i]<total[i]-1:q[i]+=1;remaining-=1
    if remaining:raise ValueError('Complete QKV/Z division failed')
    z=[t-a for t,a in zip(total,q)]
    if (sum(q),sum(z),sum(y))!=(5120,3072,2560):raise ValueError('Original output extents changed')
    return dict(qkv=q,z=z,out=list(y),combined_capacities=caps,combined_capacity=maximum,
                output_row_movement=-penalty,search_radius=radius,integer=True,physical=False)


def apply_counts(region, counts):
    result=deepcopy(region)
    for index,key in ((0,'qkv'),(1,'z'),(4,'out')):
        m=result['matrices'][index];values=counts[key]
        if len(values)!=m['groups'] or sum(values)!=m['output_tiles'] or any(type(c) is not int or c<=0 for c in values):
            raise ValueError('Original complete row ownership changed')
        first=0;partitions=[]
        for group,count in enumerate(values):
            partitions.append(dict(rank_start=group*m['k_blocks'],workers=m['k_blocks'],max_k_parts=1,
                                   output_start=first,output_count=count));first+=count
        m.pop('group_rotation',None);m['group_partitions']=partitions
        m['loop_output_tile']='group.output_start + iteration'
    classes=[]
    for rank in range(3555):
        fp=sum(local_rows(m,rank) for m in result['matrices'] if m['dtype']=='F8_E4M3')
        bf=sum(local_rows(m,rank) for m in result['matrices'] if m['dtype']=='BF16')
        if classes and (fp,bf)==(classes[-1]['fp8_slots'],classes[-1]['bf16_slots']):classes[-1]['rank_end']+=1
        else:classes.append(dict(rank_start=rank,rank_end=rank+1,fp8_slots=fp,bf16_slots=bf,auxiliary_pages=0,page_prefix=0))
    result['banks']=dict(classes=classes,auxiliary_map='frontend-bank-placement.json',region_only_auxiliary_placement=False,
                         fp8_tiles=sum(m['tiles'] for m in result['matrices'] if m['dtype']=='F8_E4M3'),
                         bf16_tiles=sum(m['tiles'] for m in result['matrices'] if m['dtype']=='BF16'))
    return result


def root_setups(region):
    """Compact graph-planning descriptors; complete descriptor audit is separate."""
    records=[]
    for rank in range(0,3520,40):
        ds=[]
        for mi,m in enumerate(region['matrices']):
            key=rank%m['k_blocks'];group=rank//m['k_blocks'];count=local_rows(m,rank)
            if 'group_partitions' in m:
                first=m['group_partitions'][group]['output_start'] if group<m['groups'] else 0;stride=1
            else:first=(group-m['group_rotation'])%m['groups'];stride=m['groups']
            base=tile_owner(region,mi,first*m['k_blocks']+key)['byte_offset']//4 if count else 0
            ds.extend([base,count,first*m['tile_shape'][0],stride*m['tile_shape'][0],key,m['k_blocks'],int(m['dtype']=='BF16'),m['tile_shape'][0]])
        records.append([rank_xy(region,rank),ds])
    return records
