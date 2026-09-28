"""Bind selected real ELF costs to the P40 per-PE bank census.

Selected full-bank consumers are compiler observations. Re-expanding calibration
banks and propagating code costs to unsampled PEs are estimates, never complete
stage admission. A candidate that cannot fit its immutable matrix prefixes is
rejected before dispatch; it requires ownership/layout changes or smaller code.
"""
from collections import Counter

CEILING=48128


def audit_frontend_resources(network, profiles, dialogue, census, samples, sram):
    if not sram['passed'] or sram['application'] != [len(samples),1]:
        raise ValueError('Complete selected compiler SRAM admission required')
    original={tuple(p['pe']):p for p in profiles}
    ends={tuple(p['pe']):p['bytes'] for p in dialogue['bank_ends']}
    prefixes={tuple(p['pe']):p['bytes'] for p in dialogue['bank_prefixes']}
    margins={(c[0],c[1]):c[3] for c in census['cells']}
    if len(original)!=11388 or len(margins)!=11388 or set(original)!=set(margins):
        raise ValueError('Complete original per-PE census required')
    actual={};ox,oy=sram['offset']
    for r in sram['records']:
        if not r['passed'] or r['stack_allowance_bytes']!=4096 or r['ordinary_address_ceiling']!=CEILING:
            raise ValueError('Changed selected compiler gate')
        for x,y,w,h in r['rectangles']:
            if y!=oy or h!=1:raise ValueError('Selected compiler geometry')
            for xx in range(x,x+w):
                i=xx-ox
                if i in actual or not 0<=i<len(samples):raise ValueError('Aliased compiler coverage')
                if r['source']!=samples[i]['source']:raise ValueError('Compiled cohost source mismatch')
                actual[i]=r['low_section_end']+4096
    if set(actual)!=set(range(len(samples))):raise ValueError('Missing compiled sample')
    records=[];costs={};consumers={}
    for i,s in enumerate(samples):
        pe=tuple(s['original_pe']);p=original[pe];params=s['parameters'];words=params['bank_words']
        if (s['original_bank_words']*4!=ends[pe] or words>s['original_bank_words']
                or s['payload_changed']!=(words!=s['original_bank_words'])):
            raise ValueError('Calibration bank no longer describes original cohost')
        if {k:v for k,v in params.items() if k!='bank_words' and not k.startswith('frontend_')} != {
                k:v for k,v in p['parameters'].items() if k!='bank_words'}:
            raise ValueError('Original cohost parameters changed')
        estimate=actual[i]+4*(s['original_bank_words']-words)
        delta=estimate-(CEILING-margins[pe])
        source=s['source'];costs[source]=max(costs.get(source,0),delta)
        record=dict(pe=list(pe),source=source,compiled_bytes_with_stack=actual[i],calibration_bank_bytes=words*4,
                    original_bank_bytes=ends[pe],added_bytes_estimate=delta,full_bank_bytes_estimate=estimate,
                    actual_original_bank_retained=not s['payload_changed'])
        records.append(record)
        if 'frontend_group' in params:
            if s['payload_changed'] or params['frontend_group'] in consumers:raise ValueError('Frontend bank dropped or group aliased')
            consumers[params['frontend_group']]=record
    if set(consumers)!=set(range(16)):raise ValueError('Not all original frontend groups compiled')
    producers={tuple(x['pe']) for x in network['producers']}
    mergers={tuple(x['pe']) for x in network['mergers']}
    consumer_pes={tuple(x['pe']) for x in network['consumers']}
    reservations=[];excesses=[];prefix_failures=[]
    for pe in sorted(producers|mergers|consumer_pes):
        flags=''.join(str(int(v)) for v in (pe in producers,pe in mergers,pe in consumer_pes))
        source='frontend_'+flags+'_'+original[pe]['source']
        if source not in costs:raise ValueError('Uncalibrated composition variant')
        # Preserve128B for unsampled code/alignment variation. This is an
        # explicit planning guard, not a replacement for a full ELF census.
        cost=costs[source]+128
        limit=ends[pe]+margins[pe]-cost
        row=dict(pe=list(pe),source=source,reserved_extra_bytes=cost,estimated_bank_limit=limit,
                 current_bank_bytes=ends[pe],matrix_prefix_bytes=prefixes[pe])
        reservations.append(row)
        if limit<ends[pe]:excesses.append(row)
        if limit<prefixes[pe]:prefix_failures.append(row)
    return dict(schema='frontend-cohost-resource-audit-v1',passed=True,
                selected_compiler_passed=True,selected_samples=len(samples),
                all16_frontend_consumers_keep_complete_original_banks=True,
                actual_max_consumer_bytes_with_stack=max(c['compiled_bytes_with_stack'] for c in consumers.values()),
                actual_min_consumer_margin=min(CEILING-c['compiled_bytes_with_stack'] for c in consumers.values()),
                selected_records=records,estimated_reservations=reservations,
                fixed_bank_candidate_accepted=not excesses,estimated_overfull_pes=len(excesses),
                estimated_matrix_prefix_conflicts=len(prefix_failures),
                total_extra_reservation_bytes=sum(r['reserved_extra_bytes'] for r in reservations),
                maximum_estimated_prefix_excess=max((r['matrix_prefix_bytes']-r['estimated_bank_limit'] for r in prefix_failures),default=0),
                bank_relocation_only_candidate_accepted=not prefix_failures,
                compile_scope='Only the selected cohosts; most have explicit calibration banks.',
                estimates_are_full_compile_admission=False,whole_stage_compiled=False,physical=False,
                numerical_qualified=False,complete_neural_stage=False,complete_model_speed=False)
