"""Lossless matrix/network co-placement with explicit retained page identities.

Budgets are measured-code estimates. Only a subsequent complete ELF census can
admit the composed stage. Matrix scale aliases require remote source-bit checks.
"""
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy

from spatial.compact_mixer import CompactMixerPlacement
from spatial.mixer_rebalance import solve_counts, apply_counts, root_setups
from spatial.mixer_frontend_network import lower_frontend_network


def measured_costs(samples, sram, margins):
    actual = {}
    ox, oy = sram['offset']
    for record in sram['records']:
        if record['stack_allowance_bytes'] != 4096 or record['ordinary_address_ceiling'] != 48128:
            raise ValueError('Changed calibration SRAM gate')
        for x, y, width, height in record['rectangles']:
            if y != oy or height != 1:
                raise ValueError('Calibration geometry')
            for xx in range(x, x+width):
                i = xx-ox
                if i in actual or not 0 <= i < len(samples) or record['source'] != samples[i]['source']:
                    raise ValueError('Calibration coverage or source identity')
                actual[i] = record['low_section_end']+4096
    if set(actual) != set(range(len(samples))):
        raise ValueError('Incomplete calibration census')
    costs = defaultdict(int)
    for i, sample in enumerate(samples):
        params = sample['parameters']
        delta = actual[i]+4*(sample['original_bank_words']-params['bank_words'])-(48128-margins[tuple(sample['original_pe'])])
        if 'frontend_group' in params:
            key = 'consumer'
        elif 'frontend_horizontal0' in params:
            ports = sum(params[k] for k in ('frontend_horizontal0','frontend_horizontal1','frontend_horizontal2','frontend_vertical'))
            key = ('root_merge_' if 'frontend_source_color' in params else 'merge_')+str(ports)
        else:
            key = 'root'
        costs[key] = max(costs[key], delta)
    return dict(costs), actual


def plan_frontend_stage(region, profiles, dialogue, census, samples, sram, compact_sram, *, columns=tuple(range(86,102))):
    margins = {(c[0],c[1]):c[3] for c in census['cells']}
    ends = {tuple(p['pe']):p['bytes'] for p in dialogue['bank_ends']}
    if set(margins) != set(ends) or set(ends) != {tuple(p['pe']) for p in profiles}:
        raise ValueError('Complete qualified stage required')
    costs, before = measured_costs(samples, sram, margins)
    _, after = measured_costs(samples, compact_sram, margins)
    compact_delta = max(after[i]-before[i] for i in before)
    if compact_delta > 128:
        raise ValueError('Compact reader exceeds planning reservation')
    base = {pe:ends[pe]+margins[pe]-16 for pe in ends}
    for pe in base:
        if 63 <= pe[0] < 108 and 67 <= pe[1] < 146 and pe != (107,145):
            base[pe] -= 128
    roots = {tuple(pe) for pe,_ in root_setups(region)}
    initial = dict(base)
    for pe in roots:
        initial[pe] -= costs['root']+128
    for x in columns:
        pe = (x,145)
        initial[pe] = ends[pe]+margins[pe]-costs['consumer']-384
    reserved = {}; limits = dict(initial); history = []
    for iteration in range(12):
        counts = solve_counts(region, limits, compact_scales=True)
        candidate = apply_counts(region, counts)
        network = lower_frontend_network(candidate, profiles, root_setups(candidate), columns=columns,
                                         prune_unused=True, native_frames=True, batch_rows=True)
        next_reserved = dict(reserved)
        for merger in network['mergers']:
            pe = tuple(merger['pe']); ports = len(merger['horizontal'])+int(merger['vertical'])
            # One-port code is conservatively covered by the two-port sample.
            cost = costs['merge_'+str(max(2,ports))]+128+(costs['root'] if pe in roots else 0)
            next_reserved[pe] = max(reserved.get(pe,0),cost)
        history.append(dict(iteration=iteration,capacity=counts['combined_capacity'],mergers=len(network['mergers']),reserved=len(next_reserved)))
        if next_reserved == reserved:
            break
        reserved = next_reserved
        limits = dict(initial)
        for pe,cost in reserved.items(): limits[pe] = base[pe]-cost
    else:
        raise ValueError('Bounded monotone placement did not converge')
    return dict(schema='frontend-weight-network-plan-v1',region=candidate,counts=counts,network=network,
                columns=list(columns),history=history,measured_native_costs=costs,measured_compact_max_delta=compact_delta,
                reserved_mergers=[dict(pe=list(pe),bytes=cost) for pe,cost in sorted(reserved.items())],
                limits=[dict(pe=list(pe),bytes=end) for pe,end in sorted(limits.items())],
                compiled=False,physical=False,numerical_qualified=False)


def lower_frontend_banks(dialogue, placement):
    compact = CompactMixerPlacement(placement['region'])
    prefixes = {tuple(p['pe']):p['bytes'] for p in dialogue['bank_prefixes']}
    old_prefixes = dict(prefixes)
    prefixes.update({tuple(r['pe']):r['bytes'] for r in compact.records.values() if tuple(r['pe']) in prefixes})
    limits = {tuple(p['pe']):p['bytes'] for p in placement['limits']}
    capacities = {pe:(limits[pe]-size)//128 for pe,size in prefixes.items()}
    if min(capacities.values()) < 0:
        raise ValueError('An immutable compact matrix prefix exceeds its reserved bank')
    local = defaultdict(list); source = {}
    for span in dialogue['spans']:
        for j in range(span['pages']):
            page = span['page_start']+j; pe = tuple(span['pe'])
            local[pe].append((span['byte_offset']+128*j,page)); source[page] = pe
    owners = {}; spills = []
    for pe,values in sorted(local.items()):
        for i,(_,page) in enumerate(sorted(values)):
            if i < capacities[pe]: owners[page] = pe
            else: spills.append(page)
        capacities[pe] -= min(len(values),capacities[pe])
    free = {pe for pe,n in capacities.items() if n}
    for page in spills:
        if not free: raise ValueError('Complete retained dialogue state has no bank capacity')
        x,y = source[page]
        pe = min(free,key=lambda p:(abs(p[0]-x)+abs(p[1]-y),p))
        owners[page] = pe; capacities[pe] -= 1
        if not capacities[pe]: free.remove(pe)
    cursor = dict(prefixes); spans = []
    for page,pe in sorted(owners.items()):
        offset = cursor[pe]; cursor[pe] += 128
        if spans and spans[-1]['pe']==list(pe) and spans[-1]['page_start']+spans[-1]['pages']==page and spans[-1]['byte_offset']+spans[-1]['pages']*128==offset:
            spans[-1]['pages'] += 1
        else: spans.append(dict(page_start=page,pages=1,pe=list(pe),byte_offset=offset))
    ends = {tuple(p['pe']):p['bytes'] for p in dialogue['bank_ends']}; ends.update(cursor)
    plan = dict(schema='frontend-compact-layer-banks-v1',stage=dialogue['stage'],region=dialogue['region'],
                conversations=1,work_slots=2,retained_source_request=0,objects=deepcopy(dialogue['objects']),
                source_logical_pages=dialogue['source_logical_pages'],retained_pages=dialogue['retained_pages'],
                source_allocated_bytes=dialogue['allocated_bytes'],allocated_bytes=sum(ends.values()),
                scale_alias_bytes=sum(old_prefixes.values())-sum(prefixes.values()),
                retained_pages_changing_pe=sum(pe!=source[page] for page,pe in owners.items()),
                bank_prefixes=[dict(pe=list(pe),bytes=size) for pe,size in sorted(prefixes.items())],
                bank_ends=[dict(pe=list(pe),bytes=size) for pe,size in sorted(ends.items())],spans=spans,
                compiled=False,physical=False,source_alias_values_verified=False)
    audit_frontend_banks(dialogue,placement,plan)
    return plan


def audit_frontend_banks(dialogue, placement, plan):
    compact = CompactMixerPlacement(placement['region'])
    expected_prefix = {tuple(p['pe']):p['bytes'] for p in dialogue['bank_prefixes']}
    expected_prefix.update({tuple(r['pe']):r['bytes'] for r in compact.records.values() if tuple(r['pe']) in expected_prefix})
    prefixes = {tuple(p['pe']):p['bytes'] for p in plan['bank_prefixes']}
    ends = {tuple(p['pe']):p['bytes'] for p in plan['bank_ends']}
    limits = {tuple(p['pe']):p['bytes'] for p in placement['limits']}
    expected = {s['page_start']+j for s in dialogue['spans'] for j in range(s['pages'])}
    if (plan['schema']!='frontend-compact-layer-banks-v1' or plan['objects']!=dialogue['objects']
            or (plan['conversations'],plan['work_slots'],plan['retained_source_request'])!=(1,2,0)
            or prefixes!=expected_prefix or len(prefixes)!=len(plan['bank_prefixes'])
            or len(ends)!=len(plan['bank_ends']) or set(ends)!=set(limits)):
        raise ValueError('Original bank/page identity changed')
    seen=set(); intervals=defaultdict(list); last=0
    for s in plan['spans']:
        pe=tuple(s['pe']); start=s['page_start']; count=s['pages']; offset=s['byte_offset']
        if start<last or count<=0 or pe not in prefixes or offset<prefixes[pe] or (offset-prefixes[pe])%128:
            raise ValueError('Invalid retained page interval')
        for page in range(start,start+count):
            if page not in expected or page in seen: raise ValueError('Removed or duplicate dialogue page')
            seen.add(page)
        intervals[pe].append((offset,offset+128*count));last=start+count
    if seen!=expected: raise ValueError('Missing original retained pages')
    for pe,end in ends.items():
        pos=prefixes.get(pe,0)
        for first,final in sorted(intervals[pe]):
            if first!=pos: raise ValueError('Physical bank gap or overlap')
            pos=final
        if pos!=end or end>limits[pe] or end%4: raise ValueError('Bank exceeds measured-code reservation')
    if (plan['retained_pages']!=len(expected) or plan['allocated_bytes']!=sum(ends.values())
            or dialogue['allocated_bytes']-sum(ends.values())!=plan['scale_alias_bytes']):
        raise ValueError('Lossless compact byte accounting')
    return dict(passed=True,pages=len(expected),allocated_bytes=sum(ends.values()),physical=False)


class FrontendAuxiliaryPlacement:
    def __init__(self, dialogue, placement, plan):
        audit_frontend_banks(dialogue,placement,plan)
        self.spans=plan['spans']; self.starts=[s['page_start'] for s in self.spans]

    def owner(self,page):
        if type(page) is not int or page<0: raise ValueError('Retained logical page required')
        i=bisect_right(self.starts,page)-1
        if i<0 or page>=self.spans[i]['page_start']+self.spans[i]['pages']:
            raise ValueError('Removed context or out-of-range page')
        s=self.spans[i]
        return dict(pe=s['pe'],byte_offset=s['byte_offset']+128*(page-s['page_start']),bytes=128)
