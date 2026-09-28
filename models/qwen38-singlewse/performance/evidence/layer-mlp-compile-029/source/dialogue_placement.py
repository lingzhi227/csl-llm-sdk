"""Specialize a qualified joint bank map for one persistent conversation.

Keep original logical page IDs, including holes belonging to the removed second
request. Compact physical tails on their existing PEs: no retained state changes
owner, no weight is removed, and the two temporary work slots stay intact.
This is storage lowering, not a stateful neural runtime or speed admission.
"""
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy

from spatial.joint_placement import audit_joint_banks, JointAuxiliaryPlacement


def _objects(original):
    objects = []
    cursor = 0
    for obj in original['auxiliary']:
        start, pages = obj['page_start'], obj['pages']
        if start != cursor or pages <= 0 or pages != (obj['bytes'] + 127) // 128:
            raise ValueError('Original auxiliary extent is not a complete page partition')
        cursor += pages
        kept = pages
        if obj['kind'] == 'request_state':
            if (obj['requests'] != 2 or obj['bytes_per_request'] % 128
                    or obj['bytes'] != 2 * obj['bytes_per_request']):
                raise ValueError('Only two complete page-aligned source contexts are supported')
            kept = obj['bytes_per_request'] // 128
        elif obj['kind'] == 'slot_value':
            if obj['slots'] != 2 or obj['bytes'] != 2 * obj['bytes_per_slot']:
                raise ValueError('Both original temporary work slots are required')
        elif obj['kind'] != 'weight':
            raise ValueError('Unknown original auxiliary object')
        objects.append(dict(id=obj['id'], kind=obj['kind'], page_start=start,
                            source_pages=pages, retained_pages=kept,
                            dropped_pages=pages-kept))
    if cursor != original['auxiliary_pages']:
        raise ValueError('Original logical page extent changed')
    return objects


def lower_dialogue_banks(original, region, source_plan, profiles):
    """Mutate only profile bank lengths after an independently checked lowering."""
    source = JointAuxiliaryPlacement(original, region, source_plan)
    objects = _objects(original)
    by_pe = {tuple(p['pe']): p for p in profiles}
    if len(by_pe) != len(profiles):
        raise ValueError('Duplicate profile owner')
    old_ends = {tuple(p['pe']): p['bytes'] for p in source_plan['bank_ends']}
    if set(by_pe) != set(old_ends) or any(
            p['parameters'].get('bank_words', 0)*4 != old_ends[pe]
            for pe, p in by_pe.items()):
        raise ValueError('Profiles do not bind the qualified source banks')
    prefixes = {tuple(p['pe']): p['bytes'] for p in source_plan['bank_prefixes']}
    local = defaultdict(list)
    for obj in objects:
        for page in range(obj['page_start'], obj['page_start'] + obj['retained_pages']):
            owner = source.owner(page)
            local[tuple(owner['pe'])].append((owner['byte_offset'], page))
    spans = []
    ends = dict(old_ends)
    for pe, base in sorted(prefixes.items()):
        cursor = base
        for _, page in sorted(local[pe]):
            if (spans and spans[-1]['pe'] == list(pe)
                    and spans[-1]['page_start'] + spans[-1]['pages'] == page
                    and spans[-1]['byte_offset'] + 128*spans[-1]['pages'] == cursor):
                spans[-1]['pages'] += 1
            else:
                spans.append(dict(page_start=page, pages=1, pe=list(pe), byte_offset=cursor))
            cursor += 128
        ends[pe] = cursor
    plan = dict(schema='single-dialogue-layer-banks-v1', stage=source_plan['stage'],
                region=region['id'], conversations=1, retained_source_request=0,
                work_slots=2, context_changed=False, source_logical_pages=original['auxiliary_pages'],
                retained_pages=sum(o['retained_pages'] for o in objects),
                removed_state_pages=sum(o['dropped_pages'] for o in objects),
                source_allocated_bytes=sum(old_ends.values()), allocated_bytes=sum(ends.values()),
                reclaimed_bytes=sum(old_ends.values())-sum(ends.values()),
                objects=objects, spans=sorted(spans, key=lambda s: s['page_start']),
                bank_prefixes=deepcopy(source_plan['bank_prefixes']),
                bank_ends=[dict(pe=list(pe), bytes=end) for pe, end in sorted(ends.items())],
                retained_pages_changing_pe=0, compiled_sram_qualified=False,
                physical=False, neural_execution=False, multi_turn_qualified=False)
    audit_dialogue_banks(original, region, source_plan, plan)
    for pe, profile in by_pe.items():
        if 'bank_words' in profile['parameters']:
            profile['parameters']['bank_words'] = ends[pe] // 4
    return plan


def audit_dialogue_banks(original, region, source_plan, plan):
    audit_joint_banks(original, region, source_plan)
    expected_objects = _objects(original)
    if (plan['schema'] != 'single-dialogue-layer-banks-v1'
            or plan['objects'] != expected_objects
            or (plan['conversations'], plan['retained_source_request'], plan['work_slots']) != (1, 0, 2)
            or plan['context_changed'] is not False
            or plan['stage'] != source_plan['stage'] or plan['region'] != region['id']):
        raise ValueError('Original weights, work slots, or retained conversation changed')
    prefixes = {tuple(p['pe']): p['bytes'] for p in plan['bank_prefixes']}
    ends = {tuple(p['pe']): p['bytes'] for p in plan['bank_ends']}
    old_ends = {tuple(p['pe']): p['bytes'] for p in source_plan['bank_ends']}
    if (plan['bank_prefixes'] != source_plan['bank_prefixes'] or len(ends) != len(plan['bank_ends'])
            or set(ends) != set(old_ends) or any(end < 0 or end % 4 or end > old_ends[pe] for pe, end in ends.items())):
        raise ValueError('Original bank prefix or bank ownership changed')
    expected = bytearray(original['auxiliary_pages'])
    for obj in expected_objects:
        start, count = obj['page_start'], obj['retained_pages']
        expected[start:start+count] = b'\x01' * count
    observed = bytearray(len(expected))
    physical = defaultdict(list)
    source = JointAuxiliaryPlacement(original, region, source_plan)
    last_end = 0
    for span in plan['spans']:
        start, count = span['page_start'], span['pages']
        pe, offset = tuple(span['pe']), span['byte_offset']
        if (start < last_end or count <= 0 or start+count > len(expected)
                or pe not in prefixes or offset < prefixes[pe] or (offset-prefixes[pe]) % 128
                or offset+128*count > ends[pe]):
            raise ValueError('Invalid or overlapping retained page span')
        for page in range(start, start+count):
            if not expected[page] or observed[page] or tuple(source.owner(page)['pe']) != pe:
                raise ValueError('Retained page is missing, duplicated, relocated, or belongs to removed context')
            observed[page] = 1
        physical[pe].append((offset, offset+128*count))
        last_end = start+count
    if observed != expected:
        raise ValueError('Retained original page coverage is incomplete')
    for pe, end in ends.items():
        cursor = prefixes.get(pe, old_ends[pe])
        for first, final in sorted(physical[pe]):
            if first != cursor:
                raise ValueError('Physical bank alias or unaccounted bytes')
            cursor = final
        if cursor != end:
            raise ValueError('Unaccounted bank tail')
    retained = sum(expected)
    reclaimed = 128*(len(expected)-retained)
    if (plan['source_logical_pages'] != len(expected) or plan['retained_pages'] != retained
            or plan['removed_state_pages'] != len(expected)-retained
            or plan['source_allocated_bytes'] != sum(old_ends.values())
            or plan['allocated_bytes'] != sum(ends.values()) or plan['reclaimed_bytes'] != reclaimed
            or sum(old_ends.values())-sum(ends.values()) != reclaimed
            or plan['retained_pages_changing_pe'] != 0):
        raise ValueError('Single-conversation storage accounting mismatch')
    return dict(passed=True, retained_pages=retained, removed_state_pages=len(expected)-retained,
                reclaimed_bytes=reclaimed, physical=False, neural_execution=False)


class DialogueAuxiliaryPlacement:
    """Resolve only retained pages; stale references to the other request fail."""
    def __init__(self, original, region, source_plan, plan):
        audit_dialogue_banks(original, region, source_plan, plan)
        self.original = original
        self.spans = plan['spans']
        self.starts = [s['page_start'] for s in self.spans]

    def owner(self, page):
        if type(page) is not int or page < 0:
            raise ValueError('Retained logical page required')
        index = bisect_right(self.starts, page)-1
        if index < 0:
            raise ValueError('Retained logical page required')
        span = self.spans[index]
        if page >= span['page_start']+span['pages']:
            raise ValueError('Removed context or out-of-range page')
        return dict(pe=span['pe'], byte_offset=span['byte_offset']+128*(page-span['page_start']), bytes=128)

    def gdn_owner(self, request, head, key_block, value_block):
        state = next(a for a in self.original['auxiliary'] if a['id'] == 'recurrent.0')
        if request != 0 or not 0 <= head < 48 or not 0 <= key_block < 32 or not 0 <= value_block < 16:
            raise ValueError('Single retained conversation GDN coordinate required')
        page = state['page_start']+(head*32+key_block)*16+value_block
        return dict(**self.owner(page), state=state['id'], request=0, head=head,
                    key_start=key_block*4, value_start=value_block*8, shape=[4, 8])
