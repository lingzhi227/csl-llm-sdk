"""Joint complete MLP row ownership and full-key recurrent state placement.

This is a measured-cost planning gate, not a compiled or routed admission.
All original matrix elements and all auxiliary namespaces remain present. No
MLP K cohort/tree changes; only contiguous output-row intervals are rebalanced.
"""
from collections import defaultdict
from copy import deepcopy
from spatial.compact_mlp import CompactMlpPlacement
from spatial.layer_schedule import auxiliary_owner, local_rows, rank_xy
from spatial.mixer_rebalance import balanced_counts

STATE_FIRST = 1684
STATE_PAGES = 24576
STATE_BYTES = 48*128*128*4


def ordinary(profile):
    p = profile['parameters']
    return (profile['source']=='compact_layer_projection.csl' and p['max_parts']==1
            and not p['root'] and not p.get('fusion_actor',False) and not p.get('mlp_sender',False))


def apply_counts(region, counts):
    result=deepcopy(region); groups=result['matrices'][0]['group_partitions']
    if len(counts)!=len(groups) or sum(counts)!=result['matrices'][0]['output_tiles'] or any(type(n) is not int or n<=0 for n in counts):
        raise ValueError('Complete original MLP output extent')
    first=0
    for group,count in zip(groups,counts):
        group.update(output_start=first,output_count=count);first+=count
    for matrix in result['matrices']:matrix['group_partitions']=deepcopy(groups)
    # Original region-local auxiliary pages become explicit namespaced spans.
    # The native reader still uses these classes to reconstruct matrix offsets.
    classes=[]
    for rank in range(result['rect'][2]*result['rect'][3]):
        fp=sum(local_rows(m,rank) for m in result['matrices'])
        if classes and classes[-1]['fp8_slots']==fp:classes[-1]['rank_end']+=1
        else:classes.append(dict(rank_start=rank,rank_end=rank+1,fp8_slots=fp,bf16_slots=0,auxiliary_pages=0,page_prefix=0))
    result['banks']=dict(classes=classes,fp8_tiles=sum(m['tiles'] for m in result['matrices']),bf16_tiles=0,
                         auxiliary_map='gdn-bank-placement.json',region_only_auxiliary_placement=False)
    # Physical actor eligibility remains tied to the actual new matrix prefix.
    for actor in result.get('quantization_actors',[]):
        actor['resident_payload_bytes']=sum(local_rows(m,actor['rank']) for m in result['matrices'])*260
    return result


def plan(stage, banks, census, profiles, *, mlp_cost=1632, mixer_cost=1728, guard=16):
    if stage['id']!='layer_00' or banks['conversations']!=1 or banks['work_slots']!=2:
        raise ValueError('Pinned complete single-dialogue layer required')
    margins={tuple(c[:2]):c[3] for c in census['cells']}
    ends={tuple(r['pe']):r['bytes'] for r in banks['bank_ends']}
    prefixes={tuple(r['pe']):r['bytes'] for r in banks['bank_prefixes']}
    prof={tuple(p['pe']):p for p in profiles}
    if set(margins)!=set(ends) or set(prof)!=set(ends):raise ValueError('Complete measured stage required')
    old=CompactMlpPlacement(stage);reserved={}; selected=[]; counts={};capacity={}
    # Three full gate/up K cohorts and the lowest full down cohort. This keeps
    # their K reductions fixed and avoids native root/fusion/granted-bus leases.
    cohorts={'gate_up':(29,30,31),'down':(8,)}
    for region in stage['regions']:
        role=region['role']
        if role not in cohorts:continue
        caps=[]
        for index,group in enumerate(region['matrices'][0]['group_partitions']):
            local=[]
            for rank in range(group['rank_start'],group['rank_start']+group['workers']):
                pe=tuple(rank_xy(region,rank));r=old.records[pe]
                if index in cohorts[role] and ordinary(prof[pe]):
                    selected.append(dict(pe=list(pe),columns=8,role=role));reserved[pe]=mlp_cost
                budget=ends[pe]+margins[pe]-guard-reserved.get(pe,0)-(4096 if pe in reserved else 0)
                def size(n):
                    # Worst starting128-row alignment; exact tables are checked
                    # after contiguous row intervals have been chosen.
                    return (n*64+(127+n*r['rows']+127)//128)*r['parts']*r['branches']*4
                local.append(max([0]+[n for n in range(1,256) if size(n)<=budget]))
            caps.append(min(local))
        capacity[role]=caps
        counts[role]=balanced_counts(region['matrices'][0]['output_tiles'],caps)
    remaining=6144-sum(w['columns'] for w in selected)
    if remaining!=160:raise ValueError('Changed cohort worker eligibility requires a new plan')
    candidates=[]
    for pe,p in prof.items():
        if p['source']!='device_mixer_layer_mlp_standby.csl':continue
        available=ends[pe]+margins[pe]-prefixes[pe]-mixer_cost-guard
        if available>=32*512:candidates.append(pe)
    candidates.sort(key=lambda p:(min(abs(p[0]-x)+abs(p[1]-145) for x in range(86,102)),p))
    if len(candidates)<5:raise ValueError('Insufficient retained mixer cohost columns')
    for pe in candidates[:5]:selected.append(dict(pe=list(pe),columns=32,role='mix'));reserved[pe]=mixer_cost
    if sum(w['columns'] for w in selected)!=6144:raise ValueError('Complete full-key recurrent extent')
    # Assign each head contiguous, nonoverlapping value intervals. Larger slices
    # go first; every worker is within its own head. Distance is only a tie cost,
    # not a route or latency claim.
    first=[0]*48
    for w in sorted(selected,key=lambda w:(-w['columns'],w['pe'])):
        x,y=w['pe'];width=w['columns']
        h=min((h for h in range(48) if first[h]+width<=128),
              key=lambda h:(abs(x-(86+h//3))+abs(y-145),first[h],h))
        w.update(head=h,first=first[h]);first[h]+=width
    if first!=[128]*48:raise ValueError('Missing original head values')
    candidate=deepcopy(stage)
    candidate['regions']=[apply_counts(r,counts[r['role']]) if r['role'] in counts else r for r in stage['regions']]
    compact=CompactMlpPlacement(candidate)
    new_prefix=dict(prefixes)
    for pe,r in compact.records.items():new_prefix[pe]=r['bytes']
    limits={pe:ends[pe]+margins[pe]-guard-reserved.get(pe,0) for pe in ends}
    cursor=dict(new_prefix)
    for w in selected:
        pe=tuple(w['pe']);w['state_word']=cursor[pe]//4;cursor[pe]+=w['columns']*512
    if any(cursor[pe]>limits[pe] for pe in cursor):raise ValueError('Exact compact prefix/state exceeds reservation')
    pages={}
    for s in banks['spans']:
        for j in range(s['pages']):
            page=s['page_start']+j
            if STATE_FIRST<=page<STATE_FIRST+STATE_PAGES:continue
            pages['mix',page]=dict(pe=s['pe'],byte_offset=s['byte_offset']+128*j)
    for region in stage['regions']:
        if region['role'] not in counts:continue
        for page in range(region['auxiliary_pages']):
            original=auxiliary_owner(region,page);pe=tuple(original['pe']);r=old.records[pe]
            offset=original['byte_offset']-(r['original_bytes']-r['bytes'])
            if not r['bytes']<=offset< prefixes[pe]:raise ValueError('Original local auxiliary prefix mismatch')
            pages[region['role'],page]=dict(pe=list(pe),byte_offset=offset)
    owners={};spills=[];free={pe:(limits[pe]-cursor[pe])//128 for pe in cursor}
    for key,source in sorted(pages.items()):
        pe=tuple(source['pe'])
        if free[pe]:owners[key]=pe;free[pe]-=1
        else:spills.append(key)
    available={pe for pe,n in free.items() if n}
    for key in spills:
        if not available:raise ValueError('Retained auxiliary namespaces exceed full bank budget')
        x,y=pages[key]['pe'];pe=min(available,key=lambda p:(abs(x-p[0])+abs(y-p[1]),p))
        owners[key]=pe;free[pe]-=1
        if not free[pe]:available.remove(pe)
    spans=[]
    for (role,page),pe in sorted(owners.items()):
        offset=cursor[pe];cursor[pe]+=128
        if spans and spans[-1]['namespace']==role and spans[-1]['pe']==list(pe) and spans[-1]['page_start']+spans[-1]['pages']==page and spans[-1]['byte_offset']+128*spans[-1]['pages']==offset:spans[-1]['pages']+=1
        else:spans.append(dict(namespace=role,page_start=page,pages=1,pe=list(pe),byte_offset=offset))
    result=dict(schema='gdn-full-key-co-placement-v1',source='layer-mlp-compile-024',stage=candidate,
        counts=counts,capacities=capacity,workers=selected,spans=spans,
        source_objects={'mix':deepcopy(banks['objects']),**{r['role']:deepcopy(r['auxiliary']) for r in stage['regions'] if r['role'] in counts}},
        bank_prefixes=[dict(pe=list(pe),bytes=n) for pe,n in sorted(new_prefix.items())],
        bank_ends=[dict(pe=list(pe),bytes=n) for pe,n in sorted(cursor.items())],
        limits=[dict(pe=list(pe),bytes=n) for pe,n in sorted(limits.items())],
        source_allocated_bytes=banks['allocated_bytes'],allocated_bytes=sum(cursor.values()),
        recurrent_elements=48*128*128,auxiliary_pages=len(pages),conversations=1,work_slots=2,
        moved_auxiliary_pages=sum(owners[key]!=tuple(pages[key]['pe']) for key in pages),
        reservations=dict(mlp_port_bytes=mlp_cost,mixer_port_bytes=mixer_cost,guard_bytes=guard),
        compiled=False,routed=False,numerical_qualified=False,physical=False)
    result['audit']=audit(stage,banks,result)
    return result


def audit(original, banks, result):
    if (result['conversations'],result['work_slots'],result['recurrent_elements'])!=(1,2,786432):raise ValueError('Persistent dialogue state contract changed')
    candidate=result['stage'];compact=CompactMlpPlacement(candidate)
    old={r['role']:r for r in original['regions']};new={r['role']:r for r in candidate['regions']}
    for role in ('gate_up','down'):
        for a,b in zip(old[role]['matrices'],new[role]['matrices']):
            if {k:v for k,v in a.items() if k!='group_partitions'}!={k:v for k,v in b.items() if k!='group_partitions'}:raise ValueError('Original matrix identity changed')
            first=0
            for ga,gb in zip(a['group_partitions'],b['group_partitions']):
                if {k:v for k,v in ga.items() if k not in ('output_start','output_count')}!={k:v for k,v in gb.items() if k not in ('output_start','output_count')}:raise ValueError('Original K cohort changed')
                if gb['output_start']!=first or gb['output_count']<=0:raise ValueError('Missing/overlapping original rows')
                first+=gb['output_count']
            if first!=a['output_tiles']:raise ValueError('Incomplete original output rows')
    if new['mix']!=old['mix']:raise ValueError('Original mixer changed')
    prefixes={tuple(p['pe']):p['bytes'] for p in result['bank_prefixes']};ends={tuple(p['pe']):p['bytes'] for p in result['bank_ends']};limits={tuple(p['pe']):p['bytes'] for p in result['limits']}
    expected_prefix={tuple(p['pe']):p['bytes'] for p in banks['bank_prefixes']}
    expected_prefix.update({pe:r['bytes'] for pe,r in compact.records.items()})
    if prefixes!=expected_prefix or len(prefixes)!=len(result['bank_prefixes']) or len(ends)!=len(result['bank_ends']) or set(ends)!=set(prefixes) or set(limits)-set(ends)!={tuple(original['request_controller']['pe'])}:raise ValueError('Complete bank prefix/PE identity changed')
    expected={('mix',s['page_start']+j) for s in banks['spans'] for j in range(s['pages']) if not STATE_FIRST<=s['page_start']+j<STATE_FIRST+STATE_PAGES}
    expected|={(role,p) for role in ('gate_up','down') for p in range(old[role]['auxiliary_pages'])}
    observed=set();intervals=defaultdict(list);heads=defaultdict(list);worker_pes=set()
    for w in result['workers']:
        pe=tuple(w['pe']);start=w['state_word']*4;stop=start+w['columns']*512
        if pe in worker_pes or w['columns']<=0 or not 0<=w['head']<48 or not 0<=w['first']<w['first']+w['columns']<=128:raise ValueError('Invalid recurrent worker')
        worker_pes.add(pe);intervals[pe].append((start,stop));heads[w['head']].append((w['first'],w['first']+w['columns']))
    for h in range(48):
        cursor=0
        for first,last in sorted(heads[h]):
            if first!=cursor:raise ValueError('Recurrent value overlap/gap')
            cursor=last
        if cursor!=128:raise ValueError('Incomplete recurrent head')
    for s in result['spans']:
        for p in range(s['page_start'],s['page_start']+s['pages']):
            key=s['namespace'],p
            if key in observed or key not in expected:raise ValueError('Removed/duplicate auxiliary page')
            observed.add(key)
        intervals[tuple(s['pe'])].append((s['byte_offset'],s['byte_offset']+s['pages']*128))
    if observed!=expected:raise ValueError('Missing original auxiliary pages')
    for pe,n in prefixes.items():
        if pe in compact.records and n!=compact.records[pe]['bytes']:raise ValueError('Compact matrix prefix differs')
        cursor=n
        for first,last in sorted(intervals[pe]):
            if first!=cursor or last<=first:raise ValueError('Bank overlap/gap')
            cursor=last
        if cursor!=ends[pe] or cursor>limits[pe]:raise ValueError('Bank end/reservation mismatch')
    original_compact=CompactMlpPlacement(original)
    delta=sum(r['bytes'] for r in compact.records.values())-sum(r['bytes'] for r in original_compact.records.values())
    if sum(ends.values())!=result['allocated_bytes'] or result['allocated_bytes']!=banks['allocated_bytes']+delta:raise ValueError('Original byte conservation')
    return dict(passed=True,original_mlp_tiles=sum(m['tiles'] for role in ('gate_up','down') for m in old[role]['matrices']),
        unchanged_k_cohorts=True,recurrent_elements=786432,auxiliary_pages=len(expected),
        bank_bytes=result['allocated_bytes'],changed_scale_table_bytes=delta,workers=len(worker_pes),
        minimum_reserved_margin=min(limits[pe]-ends[pe] for pe in ends),source_word_values_verified=False,
        scope='Integer complete-address proof under measured-code reservations; requires actual route composition, full ELF census and remote original-value proof.')
