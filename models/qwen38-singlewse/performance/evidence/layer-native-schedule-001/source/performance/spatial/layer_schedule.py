"""Executable-address lowering inside each resident spatial layer region.

Each matrix uses complete K128 contraction groups. A PE's K block is invariant
over that matrix's row loop, so it receives/retains one operand packet and streams
one result pair at a time. No full per-PE array of partial projection outputs.
Layer rectangles remain disjoint; neither banks nor arithmetic cross layers.
"""
from array import array
from collections import Counter
from copy import deepcopy
from math import ceil
from statistics import median


def local_rows(matrix,rank):
    if 'group_partitions' in matrix:
        for g in matrix['group_partitions']:
            local=rank-g['rank_start']
            if 0<=local<g['workers']:
                return g['output_count']*len(range(local,matrix['k_blocks'],g['workers']))
        return 0
    if not 0<=rank<matrix['workers']:return 0
    group=rank//matrix['k_blocks']
    row=(group-matrix['group_rotation'])%matrix['groups']
    return max(0,ceil((matrix['output_tiles']-row)/matrix['groups']))


def lower_region(region,limit=35256):
    result=deepcopy(region);x,y,width,height=result['rect'];pes=width*height
    fp=array('H',[0])*pes;bf=array('H',[0])*pes;rotations={};matrices=[]
    for original in region['matrices']:
        m=deepcopy(original);k=m['shape'][1]//128;groups=pes//k
        if not groups:raise ValueError('A complete K contraction cannot fit this region')
        rows=m['shape'][0]//m['tile_shape'][0];rotation=rotations.get(k,0)%groups
        if m['tensor'].endswith('.mlp.up_proj.weight'):
            rotation=next(v['group_rotation'] for v in matrices if v['tensor'].endswith('.mlp.gate_proj.weight'))
        rotations[k]=(rotation+rows%groups)%groups
        m.update(k_blocks=k,groups=groups,workers=groups*k,output_tiles=rows,
                 group_rotation=rotation,input_k_invariant=True,
                 loop_output_tile='(group - group_rotation) modulo groups + iteration * groups')
        m.pop('stream_start')
        counts=fp if m['dtype']=='F8_E4M3' else bf
        for rank in range(m['workers']):counts[rank]+=local_rows(m,rank)
        matrices.append(m)
    classes=[];cursor=0;max_matrix=0
    for rank in range(pes):
        used=fp[rank]*260+bf[rank]*256;spare=(limit-used)//128
        if spare<0:raise ValueError('Native-loop bank exceeds region payload ceiling')
        profile=(fp[rank],bf[rank],spare)
        if classes and profile==tuple(classes[-1][k] for k in ('fp8_slots','bf16_slots','auxiliary_pages')):
            classes[-1]['rank_end']+=1
        else:classes.append(dict(rank_start=rank,rank_end=rank+1,fp8_slots=fp[rank],bf16_slots=bf[rank],
                                 auxiliary_pages=spare,page_prefix=cursor))
        cursor+=spare;max_matrix=max(max_matrix,used)
    admitted=cursor>=region['auxiliary_pages']
    result.update(matrices=matrices,banks=dict(classes=classes,matrix_payload_max=max_matrix,
                  auxiliary_page_capacity=cursor,auxiliary_pages=region['auxiliary_pages'],
                  fp8_tiles=sum(fp),bf16_tiles=sum(bf),admitted=admitted),
                  physical_rank_order='serpentine rows within this region',
                  input_distribution='Retain the one selected K128 packet for all local output iterations',
                  partial_output_storage='One row/pair per invocation, released by contraction/output credit',
                  compiled_sram_qualified=False)
    return result


def rank_xy(region,rank):
    x,y,w,h=region['rect']
    if not 0<=rank<w*h:raise ValueError('Rank outside region')
    row,column=divmod(rank,w)
    return [x+(column if row%2==0 else w-1-column),y+row]


def xy_rank(region,xy):
    x,y,w,h=region['rect'];px,py=xy
    if not x<=px<x+w or not y<=py<y+h:raise ValueError('Coordinate outside region')
    row=py-y;column=px-x
    return row*w+(column if row%2==0 else w-1-column)


def tile_owner(region,matrix_index,tile):
    m=region['matrices'][matrix_index]
    if not 0<=tile<m['tiles']:raise ValueError('Original tile extent')
    row,k=divmod(tile,m['k_blocks']);fp8=m['dtype']=='F8_E4M3'
    if 'group_partitions' in m:
        group,g=next((i,g) for i,g in enumerate(m['group_partitions']) if g['output_start']<=row<g['output_start']+g['output_count'])
        local=k%g['workers'];rank=g['rank_start']+local;iteration=row-g['output_start']
        part=k//g['workers'];slot=iteration*len(range(local,m['k_blocks'],g['workers']))+part
    else:
        group=(row+m['group_rotation'])%m['groups'];rank=group*m['k_blocks']+k
        iteration=row//m['groups'];slot=iteration
    preceding=sum(local_rows(other,rank) for other in region['matrices'][:matrix_index]
                  if (other['dtype']=='F8_E4M3')==fp8)
    total_fp8=sum(local_rows(other,rank) for other in region['matrices'] if other['dtype']=='F8_E4M3')
    size=260 if fp8 else 256
    offset=(preceding+slot)*size+(0 if fp8 else total_fp8*260)
    return dict(pe=rank_xy(region,rank),rank=rank,byte_offset=offset,bytes=size,
                k_block=k,group=group,iteration=iteration,output_tile=row)


def auxiliary_owner(region,page):
    if not 0<=page<region['auxiliary_pages']:raise ValueError('Auxiliary page extent')
    for c in region['banks']['classes']:
        size=c['auxiliary_pages'];length=(c['rank_end']-c['rank_start'])*size
        if c['page_prefix']<=page<c['page_prefix']+length:
            rank,slot=divmod(page-c['page_prefix'],size);rank+=c['rank_start']
            return dict(pe=rank_xy(region,rank),rank=rank,
                        byte_offset=c['fp8_slots']*260+c['bf16_slots']*256+slot*128,bytes=128)
    raise AssertionError('Admitted auxiliary page not located')


def gdn_state_owner(region,request,head,key_block,value_block):
    """One4-key x8-value FP32 shard per128B page; all original128x128 entries.

    This layout makes the state arithmetic directly usable even when no PE has
    a contiguous4KiB spare area. Requests occupy disjoint page intervals.
    Tile order is head,key-block,value-block; inside a tile it is key,value.
    """
    state=next(a for a in region['auxiliary'] if a['kind']=='request_state' and a['id'].startswith('recurrent.'))
    if not 0<=request<state['requests'] or not 0<=head<48 or not 0<=key_block<32 or not 0<=value_block<16:
        raise ValueError('Original GDN state coordinate')
    if state['bytes_per_request']!=48*128*128*4:raise ValueError('Original FP32 GDN state size')
    page=state['page_start']+request*(48*32*16)+head*32*16+key_block*16+value_block
    owner=auxiliary_owner(region,page);owner.update(state=state['id'],request=request,head=head,
                                                  key_start=key_block*4,value_start=value_block*8,shape=[4,8])
    return owner


def lower_layers(plan,calibration=None):
    if not plan['placement_admitted']:raise ValueError('An admitted spatial stage map is required')
    result=deepcopy(plan);failed=[]
    for s in result['stages']:
        s['regions']=[lower_region(r,plan['config']['payload_per_pe']) for r in s['regions']]
        if calibration:
            s['regions']=[select_native_shape(r,calibration,plan['config']['payload_per_pe']) if r['role'] in ('gate_up','down') else r for r in s['regions']]
        actors=[]
        for r in s['regions']:
            for c in r['banks']['classes']:
                if c['fp8_slots']==c['bf16_slots']==0:
                    for rank in range(c['rank_start'],c['rank_end']):
                        prefix=c['page_prefix']+(rank-c['rank_start'])*c['auxiliary_pages']
                        if prefix>=r['auxiliary_pages']:
                            actors.append(dict(region=r['id'],rank=rank,pe=rank_xy(r,rank),weight_bytes=0,auxiliary_bytes=0))
        if not actors:raise ValueError('No proven unallocated PE for stage request controller')
        s['request_controller']=actors[-1]
        s['request_controller']['source']='runtime/layer_controller.csl'
        s['remaining_unallocated_actor_pes']=len(actors)-1
        failed.extend(r['id'] for r in s['regions'] if not r['banks']['admitted'])
    result.update(schema='wse-resident-layer-native-loop-v1',native_loop_addressing_admitted=not failed,
                  failed_regions=failed,complete_fabric_routes_admitted=False,
                  neural_executable=False,physical=False)
    result.pop('audit',None)
    return result


def shape_costs(calibration):
    if not calibration.get('physical') or not calibration.get('passed') or not calibration.get('normal_stop'):
        raise ValueError('Qualified physical calibration required')
    costs={}
    for i,shape in enumerate(calibration['shapes']):
        if not shape['vector_scales']:continue
        samples=[c['cycles_per_invocation_by_shape'][i] for c in calibration['cases'] if c['kind']==0 and c['decode_each']==1]
        costs[tuple([shape['rows'],shape['columns']])]=median(samples)
    return costs


def retile_mlp(region,rows,columns,limit):
    """Capacity-balanced contiguous output groups, including a short tail group.

    Tail workers own a fixed small set of K slices. The retained operands total
    at most128 FP16 values per PE. Gate/up share identical output ownership, so
    their reduced values can be consumed by a colocated fused epilogue.
    """
    if rows*columns!=256 or 128%rows or 128%columns:raise ValueError('Original FP8 block alignment')
    result=deepcopy(region);matrices=result['matrices'];copies=len(matrices)
    if copies not in (1,2) or any(m['dtype']!='F8_E4M3' or m['shape']!=matrices[0]['shape'] for m in matrices):
        raise ValueError('Matched MLP matrix family required')
    n=result['rect'][2]*result['rect'][3];m,k=matrices[0]['shape'];blocks=k//columns;outrows=m//rows
    groups=[];rank=0
    while rank<n:
        workers=min(blocks,n-rank)
        if ceil(blocks/workers)*columns>128:break
        parts=ceil(blocks/workers)
        groups.append(dict(rank_start=rank,workers=workers,max_k_parts=parts,
                           row_capacity=limit//(260*copies*parts)))
        rank+=workers
    if sum(g['row_capacity'] for g in groups)<outrows:return None
    # Minimize peak serialized native calls under exact weight capacity.
    quota=0
    while sum(min(g['row_capacity'],quota//(copies*g['max_k_parts'])) for g in groups)<outrows:quota+=1
    remaining=outrows;start=0
    for g in groups:
        count=min(remaining,g['row_capacity'],quota//(copies*g['max_k_parts']))
        g.update(output_start=start,output_count=count);remaining-=count;start+=count
    groups=[g for g in groups if g['output_count']]
    for matrix in matrices:
        matrix.update(tile_shape=[rows,columns],k_blocks=blocks,groups=len(groups),workers=sum(g['workers'] for g in groups),
                      output_tiles=outrows,group_partitions=deepcopy(groups),
                      loop_output_tile='group.output_start + iteration',input_k_invariant=True)
        matrix.pop('group_rotation',None)
    fp=[sum(local_rows(m,rank) for m in matrices) for rank in range(n)]
    classes=[];cursor=0
    for rank,count in enumerate(fp):
        spare=(limit-count*260)//128
        if spare<0:return None
        if classes and classes[-1]['fp8_slots']==count:
            classes[-1]['rank_end']+=1
        else:classes.append(dict(rank_start=rank,rank_end=rank+1,fp8_slots=count,bf16_slots=0,auxiliary_pages=spare,page_prefix=cursor))
        cursor+=spare
    if cursor<result['auxiliary_pages']:return None
    result['banks']=dict(classes=classes,matrix_payload_max=max(fp)*260,auxiliary_page_capacity=cursor,
                         auxiliary_pages=result['auxiliary_pages'],fp8_tiles=sum(fp),bf16_tiles=0,admitted=True)
    result['maximum_native_invocations_per_pe']=max(fp)
    result['native_shape']=[rows,columns]
    result['epilogue']='Gate/up share output roots; BF16 SiLU/multiply is colocated. Original128-value quantization may cross two neighboring output groups.' if copies==2 else 'Full-K reduction, then BF16 and retained residual'
    if copies==2:
        actors=[]
        for c in classes:
            for pe in range(c['rank_start'],c['rank_end']):
                prefix=c['page_prefix']+(pe-c['rank_start'])*c['auxiliary_pages']
                live=max(0,min(c['auxiliary_pages'],result['auxiliary_pages']-prefix))
                payload=c['fp8_slots']*260+live*128
                if payload<=28000:
                    actors.append(dict(rank=pe,pe=rank_xy(result,pe),resident_payload_bytes=payload,groups=[],
                                       native_invocations=c['fp8_slots']))
        if not actors:return None
        actors=actors[:m//128]
        for group in range(m//128):actors[group%len(actors)]['groups'].append(group)
        result['quantization_actors']=actors
        result['quantization_schedule']='Each actor grants its assigned128-value groups in increasing group order within a stage request epoch; no later request overlaps mutable scratch.'
    return result


def select_native_shape(region,calibration,limit):
    candidates=[];costs=shape_costs(calibration)
    for shape,cost in sorted(costs.items()):
        candidate=retile_mlp(region,*shape,limit)
        if candidate:
            candidate['native_busy_cycle_screen']=candidate['maximum_native_invocations_per_pe']*cost
            candidates.append(candidate)
    if not candidates:raise ValueError('No shape admits original MLP banks and auxiliary pages')
    best=min(candidates,key=lambda r:r['native_busy_cycle_screen'])
    best['candidate_screen']=[dict(shape=r['native_shape'],maximum_native_invocations_per_pe=r['maximum_native_invocations_per_pe'],
                                   native_busy_cycles=r['native_busy_cycle_screen']) for r in candidates]
    best['screen_scope']='Conditional reuse of qualified native tile cycles only; excludes reductions, communication, all other operators and new ragged-loop overhead. Not full stage latency or TPS.'
    return best


def audit_native_schedule(plan,tensors):
    """Count real original rows/K blocks and independently enumerate each PE bank."""
    regions=0;matrix_count=tiles=0;max_used=0;page_margin=None
    for stage in plan['stages']:
        controller=stage['request_controller'];cr=next(r for r in stage['regions'] if r['id']==controller['region'])
        rank=xy_rank(cr,controller['pe']);cl=next(c for c in cr['banks']['classes'] if c['rank_start']<=rank<c['rank_end'])
        prefix=cl['page_prefix']+(rank-cl['rank_start'])*cl['auxiliary_pages']
        if cl['fp8_slots'] or cl['bf16_slots'] or prefix<cr['auxiliary_pages']:
            raise ValueError('Stage controller overlaps weights or live state/value pages')
        for r in stage['regions']:
            regions+=1;n=r['rect'][2]*r['rect'][3];classes=r['banks']['classes']
            expected_fp=[0]*n;expected_bf=[0]*n
            for mi,m in enumerate(r['matrices']):
                matrix_count+=1;original=tensors[m['tensor']];rows,columns=m['tile_shape']
                fp8=original['dtype']=='F8_E4M3';mrows,k=original['shape'];blocks=k//columns
                groups=m['groups']
                if m['shape']!=[mrows,k] or m['dtype']!=original['dtype'] or rows*columns!=(256 if fp8 else 128):
                    raise ValueError('Original matrix/complete K group changed')
                if m['k_blocks']!=blocks or m['output_tiles']!=mrows//rows or m['tiles']!=mrows//rows*blocks:
                    raise ValueError('Native loop cardinality mismatch')
                destination=expected_fp if fp8 else expected_bf
                if 'group_partitions' in m:
                    next_row=next_rank=0
                    for g in m['group_partitions']:
                        if g['output_start']!=next_row or g['rank_start']!=next_rank:raise ValueError('Ragged contraction gap')
                        next_row+=g['output_count'];next_rank+=g['workers']
                        for block in range(blocks):destination[g['rank_start']+block%g['workers']]+=g['output_count']
                    if next_row!=mrows//rows or next_rank>n:raise ValueError('Ragged matrix extent')
                else:
                    counts=Counter((outrow+m['group_rotation'])%groups for outrow in range(mrows//rows))
                    for group,count in counts.items():
                        for block in range(blocks):destination[group*blocks+block]+=count
                tiles+=m['tiles']
                for tile in {0,m['tiles']-1,min(m['tiles']-1,groups*blocks)}:
                    owner=tile_owner(r,mi,tile)
                    if ('group_partitions' not in m and owner['k_block']!=owner['rank']%blocks) or xy_rank(r,owner['pe'])!=owner['rank']:
                        raise ValueError('K input is not stable at its PE')
            rank=cursor=0
            for c in classes:
                if c['rank_start']!=rank or c['page_prefix']!=cursor:raise ValueError('Native bank census gap')
                while rank<c['rank_end']:
                    used=expected_fp[rank]*260+expected_bf[rank]*256
                    capacity=(plan['config']['payload_per_pe']-used)//128
                    if [expected_fp[rank],expected_bf[rank],capacity]!=[c['fp8_slots'],c['bf16_slots'],c['auxiliary_pages']]:
                        raise ValueError('Native local weight/address census mismatch')
                    max_used=max(max_used,used);cursor+=capacity;rank+=1
            if rank!=n or cursor<r['auxiliary_pages']:raise ValueError('Native bank/auxiliary admission failed')
            if r['role']=='gate_up' and 'quantization_actors' in r:
                all_groups=[]
                for actor in r['quantization_actors']:
                    rank=xy_rank(r,actor['pe']);c=next(c for c in classes if c['rank_start']<=rank<c['rank_end'])
                    prefix=c['page_prefix']+(rank-c['rank_start'])*c['auxiliary_pages']
                    live=max(0,min(c['auxiliary_pages'],r['auxiliary_pages']-prefix))
                    payload=expected_fp[rank]*260+expected_bf[rank]*256+live*128
                    if actor['rank']!=rank or actor['resident_payload_bytes']!=payload or payload>28000:
                        raise ValueError('Fusion actor storage changed')
                    if actor['groups']!=sorted(set(actor['groups'])):raise ValueError('Actor group ordering')
                    all_groups+=actor['groups']
                if sorted(all_groups)!=list(range(136)):raise ValueError('Original group128 quantization coverage')
            margin=cursor-r['auxiliary_pages'];page_margin=margin if page_margin is None else min(page_margin,margin)
    return dict(passed=True,regions=regions,matrices=matrix_count,native_tiles=tiles,
                maximum_matrix_payload=max_used,minimum_region_spare_auxiliary_pages=page_margin,
                full_model_layer_residency=True,compiled_sram=False,executed=False)
