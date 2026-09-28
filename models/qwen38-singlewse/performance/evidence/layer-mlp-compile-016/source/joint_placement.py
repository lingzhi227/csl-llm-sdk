"""Joint full-K matrix ownership and complete original auxiliary-bank placement.

The same adjacent40/48-PE reductions remain; QKV output row counts vary across
complete40-PE groups to leave room for the real norm cohosts. All original
auxiliary pages remain inside this layer, using measured MLP slack when needed.
Calibrated bank budgets are estimates until the complete candidate is compiled.
"""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from spatial.layer_schedule import auxiliary_owner,local_rows,rank_xy,xy_rank

CEILING=48128


def matrix_bytes(region,rank):
    return sum(local_rows(m,rank)*(260 if m['dtype']=='F8_E4M3' else 256) for m in region['matrices'])


def lower_joint_banks(stage,profiles,calibration,cohost_calibration,reserve=256):
    if stage['id']!='layer_00' or calibration['application']!=stage['rect'][2:] or reserve<256:
        raise ValueError('This calibration admits only the selected original layer00 geometry')
    original=next(r for r in stage['regions'] if r['role']=='mix');region=deepcopy(original)
    by_pe={tuple(p['pe']):p for p in profiles};old_ends={pe:p['parameters'].get('bank_words',0)*4 for pe,p in by_pe.items()}
    if len(by_pe)!=len(profiles):raise ValueError('Duplicate bank owner')
    original_total=sum(old_ends.values());controller=tuple(stage['request_controller']['pe'])
    mx,my,mw,mh=region['rect'];mix={tuple(rank_xy(region,r)) for r in range(mw*mh)}
    limits={};overheads={}
    if (cohost_calibration['before'],cohost_calibration['after'])!=('layer-backend-compile-014','layer-backend-compile-017') or not cohost_calibration['device_control']:
        raise ValueError('Final compiled internal-control calibration is required')
    samples=cohost_calibration['samples'][1:]
    if len(samples)!=8 or any(s['actual_calibration_bank_bytes']!=16384 for s in samples):raise ValueError('Changed cohost calibration bank extent')
    standby={s['parameters']['mixer_can_root']:s['overhead_with_stack'] for s in samples if s['source']=='device_mixer_layer_mlp_standby.csl'}
    root_extra=standby[True]-standby[False]
    # Final selected P35 compile017 including4096 stack. Unsampled root/norm
    # combinations use an additional256B root reserve. Another256B is retained
    # for code/section alignment variation; these remain estimates, not ELF gates.
    for pe in mix-{controller}:
        p=by_pe[pe];rank=xy_rank(region,pe);root=rank%40==0 or rank%48==0
        role=p['source'].removeprefix('mixer_')
        if role in ('layer_norm_bridge.csl','layer_norm_sender.csl'):
            cost=max(s['overhead_with_stack'] for s in samples if s['source']=='device_mixer_'+role)
        elif role=='layer_mlp_standby.csl':cost=standby[False]
        else:raise ValueError('Uncalibrated mixer role')
        cost+=root_extra*root;overheads[pe]=cost;limits[pe]=(CEILING-cost-reserve)//4*4
    ox,oy=calibration['offset'];sx,sy=stage['rect'][:2]
    measured={}
    for record in calibration['records']:
        for x,y,w,h in record['rectangles']:
            for yy in range(y,y+h):
                for xx in range(x,x+w):
                    pe=(xx-ox+sx,yy-oy+sy)
                    if pe in measured:raise ValueError('Overlapping calibration ELF rectangles')
                    measured[pe]=record
    if set(measured)!=set(by_pe):raise ValueError('Calibration does not cover the complete stage')
    for pe,p in by_pe.items():
        if pe in mix:continue
        record=measured[pe]
        if p['source']!=record['source'] or p['source']!='layer_projection.csl':raise ValueError('Changed MLP calibration role')
        # Existing banks stay intact; only append pages where actual ELF slack
        # exceeds the reserve. A near-limit original PE receives no extra pages.
        limits[pe]=old_ends[pe]+max(0,(CEILING-reserve-record['bytes_with_stack'])//128)*128
    qkv=region['matrices'][0];groups=qkv['groups'];parts=qkv['k_blocks'];capacities=[]
    if groups!=88 or parts!=40 or qkv['output_tiles']!=5120:raise ValueError('Original QKV contraction geometry')
    for group in range(groups):
        room=[]
        for key in range(parts):
            rank=group*parts+key;pe=tuple(rank_xy(region,rank))
            other=matrix_bytes(region,rank)-local_rows(qkv,rank)*260
            room.append((limits[pe]-other)//260)
        capacities.append(min(room))
    if min(capacities)<=0 or sum(capacities)<qkv['output_tiles']:raise ValueError('Complete QKV cannot fit calibrated cohosts')
    quota=1
    while sum(min(c,quota) for c in capacities)<qkv['output_tiles']:quota+=1
    remaining=qkv['output_tiles'];first=0;partitions=[]
    for group,capacity in enumerate(capacities):
        count=min(capacity,quota,remaining)
        if count<=0:raise ValueError('A rebalanced group unexpectedly becomes empty')
        partitions.append(dict(rank_start=group*parts,workers=parts,max_k_parts=1,output_start=first,output_count=count))
        first+=count;remaining-=count
    assert remaining==0
    qkv.pop('group_rotation');qkv['group_partitions']=partitions;qkv['loop_output_tile']='group.output_start + iteration'
    # The generic region-only auxiliary resolver must not silently reuse the
    # original map. This candidate has an explicit stage-wide auxiliary map.
    classes=[]
    for rank in range(mw*mh):
        fp=sum(local_rows(m,rank) for m in region['matrices'] if m['dtype']=='F8_E4M3')
        bf=sum(local_rows(m,rank) for m in region['matrices'] if m['dtype']=='BF16')
        if classes and (classes[-1]['fp8_slots'],classes[-1]['bf16_slots'])==(fp,bf):classes[-1]['rank_end']+=1
        else:classes.append(dict(rank_start=rank,rank_end=rank+1,fp8_slots=fp,bf16_slots=bf,auxiliary_pages=0,page_prefix=0))
    region['banks']=dict(classes=classes,auxiliary_map='joint-bank-placement.json',region_only_auxiliary_placement=False,
                         fp8_tiles=sum(m['tiles'] for m in region['matrices'] if m['dtype']=='F8_E4M3'),
                         bf16_tiles=sum(m['tiles'] for m in region['matrices'] if m['dtype']=='BF16'))
    bases={pe:(matrix_bytes(region,xy_rank(region,pe)) if pe in mix else old_ends[pe]) for pe in limits}
    available={pe:(limits[pe]-base)//128 for pe,base in bases.items()}
    if min(available.values())<0 or sum(available.values())<original['auxiliary_pages']:
        raise ValueError('Original auxiliary capacity cannot fit the calibrated complete layer')
    initial_available=sum(available.values());used=Counter();spans=[];sources=[]
    for c in original['banks']['classes']:
        for rank in range(c['rank_start'],c['rank_end']):
            first=c['page_prefix']+(rank-c['rank_start'])*c['auxiliary_pages']
            count=max(0,min(c['auxiliary_pages'],original['auxiliary_pages']-first))
            if count:sources.append((tuple(rank_xy(original,rank)),first,count))
    def place(page,count,pe):
        if count<=0 or count>available[pe]:raise ValueError('Auxiliary allocation exceeds its actual page capacity')
        spans.append(dict(page_start=page,pages=count,pe=list(pe),byte_offset=bases[pe]+128*used[pe]))
        used[pe]+=count;available[pe]-=count
    spills=[]
    # Reserve each original owner's local pages before allocating any spill.
    for pe,page,count in sources:
        local=min(count,available.get(pe,0))
        if local:place(page,local,pe)
        if local<count:spills.append((pe,page+local,count-local))
    x0,y0,width,height=stage['rect'];moved=0;distance_sum=0;maximum_distance=0
    for source,page,count in spills:
        for distance in range(1,width+height):
            candidates=[]
            for dx in range(-distance,distance+1):
                dy=distance-abs(dx)
                for yy in ({source[1]-dy,source[1]+dy} if dy else {source[1]}):
                    pe=(source[0]+dx,yy)
                    if x0<=pe[0]<x0+width and y0<=pe[1]<y0+height and available.get(pe,0)>0:candidates.append(pe)
            for pe in sorted(candidates):
                take=min(count,available[pe]);place(page,take,pe);page+=take;count-=take
                moved+=take;distance_sum+=distance*take;maximum_distance=max(maximum_distance,distance)
                if count==0:break
            if count==0:break
        if count:raise ValueError('Nearest available resident auxiliary pages exhausted')
    for pe,p in by_pe.items():
        if pe in bases:p['parameters']['bank_words']=(bases[pe]+128*used[pe])//4
    spans.sort(key=lambda s:s['page_start'])
    plan=dict(schema='joint-layer-bank-placement-v1',region=region['id'],stage=stage['id'],original_pages=original['auxiliary_pages'],
        spans=spans,bank_prefixes=[dict(pe=list(pe),bytes=base) for pe,base in sorted(bases.items())],
        bank_ends=[dict(pe=list(pe),bytes=p['parameters'].get('bank_words',0)*4) for pe,p in sorted(by_pe.items())],
        original_allocated_bytes=original_total,qkv_group_rows=[p['output_count'] for p in partitions],
        qkv_maximum_rows=quota,capacity_pages=initial_available,unused_page_capacity=sum(available.values()),
        moved_auxiliary_pages=moved,auxiliary_page_hops=distance_sum,maximum_auxiliary_distance=maximum_distance,
        auxiliary_pages_outside_mixer=sum(s['pages'] for s in spans if tuple(s['pe']) not in mix),
        calibration_attempt=calibration['source_attempt'],calibration_sha256=calibration['source_sha256'],
        cohost_calibration='layer-backend-compile-017',extra_bytes_reserved=reserve,
        compiled_sram_qualified=False,complete_control_routes=False,executed=False,physical=False)
    audit_joint_banks(original,region,plan)
    return region,plan


def audit_joint_banks(original,region,plan):
    expected=0;by_pe={};prefixes={tuple(p['pe']):p['bytes'] for p in plan['bank_prefixes']}
    ends={tuple(p['pe']):p['bytes'] for p in plan['bank_ends']}
    if len(prefixes)!=len(plan['bank_prefixes']) or len(ends)!=len(plan['bank_ends']):raise ValueError('Duplicate bank boundary')
    for span in plan['spans']:
        if span['page_start']!=expected or span['pages']<=0:raise ValueError('Missing or duplicated original auxiliary pages')
        expected+=span['pages'];pe=tuple(span['pe']);first=span['byte_offset'];end=first+128*span['pages']
        if pe not in prefixes or first<prefixes[pe] or end>ends[pe]:raise ValueError('Auxiliary span outside admitted bank tail')
        by_pe.setdefault(pe,[]).append((first,end))
    if expected!=original['auxiliary_pages'] or expected!=plan['original_pages']:raise ValueError('Original auxiliary capacity changed')
    for pe,base in prefixes.items():
        cursor=base
        for first,end in sorted(by_pe.get(pe,[])):
            if first!=cursor:raise ValueError('Overlapping or unaccounted bank bytes')
            cursor=end
        if cursor!=ends[pe]:raise ValueError('Unaccounted bank tail')
    if sum(ends.values())!=plan['original_allocated_bytes']:raise ValueError('Original total bank capacity changed')
    for rank in range(region['rect'][2]*region['rect'][3]):
        pe=tuple(rank_xy(region,rank))
        if pe in prefixes and prefixes[pe]!=matrix_bytes(region,rank):raise ValueError('Auxiliary storage overlaps matrix payload')
    # QKV alone changes output assignment; dimensions and every original tensor
    # identity stay intact. Other four matrix row maps are byte-for-byte metadata.
    if original['matrices'][1:]!=region['matrices'][1:]:raise ValueError('Unrequested non-QKV ownership change')
    old,new=original['matrices'][0],region['matrices'][0]
    for field in ('tensor','scale_tensor','shape','tile_shape','tiles','dtype','k_blocks','workers'):
        if old[field]!=new[field]:raise ValueError('Original QKV identity or dimensions changed')
    cursor=0
    if len(new['group_partitions'])!=old['groups']:raise ValueError('Original complete K group count changed')
    for group,part in enumerate(new['group_partitions']):
        if (part['rank_start'],part['workers'],part['output_start'])!=(group*old['k_blocks'],old['k_blocks'],cursor) or part['output_count']<=0:
            raise ValueError('QKV rows or complete K groups are missing or duplicated')
        cursor+=part['output_count']
    if cursor!=old['output_tiles']:raise ValueError('Original QKV output extent changed')
    return dict(passed=True,original_auxiliary_pages=expected,original_allocated_bytes=sum(ends.values()),physical=False)


class JointAuxiliaryPlacement:
    """Explicit resolver for every original weight/state/value auxiliary page."""
    def __init__(self,original,region,plan):
        audit_joint_banks(original,region,plan);self.spans=plan['spans'];self.starts=[s['page_start'] for s in self.spans]
        self.original=original;self.count=plan['original_pages']

    def owner(self,page):
        if not 0<=page<self.count:raise ValueError('Original auxiliary page extent')
        span=self.spans[bisect_right(self.starts,page)-1]
        return dict(pe=span['pe'],byte_offset=span['byte_offset']+128*(page-span['page_start']),bytes=128)

    def gdn_owner(self,request,head,key_block,value_block):
        state=next(a for a in self.original['auxiliary'] if a['id']=='recurrent.0')
        if not 0<=request<state['requests'] or not 0<=head<48 or not 0<=key_block<32 or not 0<=value_block<16:
            raise ValueError('Original GDN state coordinate')
        page=state['page_start']+((request*48+head)*32+key_block)*16+value_block
        return dict(**self.owner(page),state=state['id'],request=request,head=head,key_start=key_block*4,value_start=value_block*8,shape=[4,8])
