"""Bounded remote-only specialization of the complete original joint fixture.

Every source word is classified as retained exactly once or unused second-context
state. Every destination word is compared after a fresh read. This fixture has
zero-initialized state; rejecting nonzero discarded state avoids accidentally
using this preparation path to destroy an active independent conversation.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from spatial.dialogue_placement import DialogueAuxiliaryPlacement
from spatial.joint_placement import JointAuxiliaryPlacement


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.monotonic()
    config = json.loads(Path('source-bank.json').read_text())
    root = Path(config['root'])
    for name, digest in config['hashes'].items():
        if sha(root/name) != digest:
            raise ValueError('Source bank identity changed: '+name)
    oldindex = json.loads((root/'bank-index.json').read_text())
    old = {tuple(r['pe']): r for r in oldindex['records']}
    source = np.load(root/'banks.npy', mmap_mode='r')
    if source.dtype != np.dtype('<u4') or source.ndim != 1:
        raise ValueError('Original bank array encoding')
    original = json.loads(Path('original-stage.json').read_text())
    joint = json.loads(Path('joint-stage.json').read_text())
    original = next(r for r in original['regions'] if r['role'] == 'mix')
    region = next(r for r in joint['regions'] if r['role'] == 'mix')
    before = json.loads(Path('joint-bank-placement.json').read_text())
    plan = json.loads(Path('dialogue-bank-placement.json').read_text())
    src = JointAuxiliaryPlacement(original, region, before)
    dst = DialogueAuxiliaryPlacement(original, region, before, plan)
    old_ends = {tuple(p['pe']): p['bytes'] for p in before['bank_ends'] if p['bytes']}
    if (len(old) != len(oldindex['records']) or set(old) != set(old_ends)
            or any(r['words']*4 != old_ends[pe] for pe, r in old.items())
            or source.nbytes != plan['source_allocated_bytes']):
        raise ValueError('Original bank index does not match the admitted source')
    cursor = 0
    for record in sorted(old.values(), key=lambda r: r['offset']):
        if record['offset'] != cursor:
            raise ValueError('Original bank index alias or gap')
        cursor += record['words']
    if cursor != source.size:
        raise ValueError('Original bank index extent')
    payload = {tuple(pe): p['parameters'].get('bank_words', 0)*4
               for p in json.loads(Path('profiles.json').read_text())['profiles'] for pe in p['pes']}
    if payload != {tuple(p['pe']): p['bytes'] for p in plan['bank_ends']}:
        raise ValueError('Compiled profile bank lengths differ from the specialized map')
    new = {}
    cursor = 0
    for pe, size in sorted(payload.items(), key=lambda item: (item[0][1], item[0][0])):
        if size:
            new[pe] = dict(pe=list(pe), words=size//4, offset=cursor)
            cursor += size//4
    if cursor*4 != plan['allocated_bytes'] or cursor*4 > 512 << 20:
        raise ValueError('Candidate bank extent')
    output = np.lib.format.open_memmap('banks.npy', mode='w+', dtype='<u4', shape=(cursor,))
    source_class = np.zeros(source.size, np.uint8)
    destination_covered = np.zeros(cursor, np.bool_)
    transfers = []

    def position(index, address):
        record = index[tuple(address['pe'])]
        offset, size = address['byte_offset'], address['bytes']
        if offset < 0 or offset % 4 or size <= 0 or size % 4 or offset+size > record['words']*4:
            raise ValueError('Transfer outside a declared bank')
        return record['offset']+offset//4, size//4

    def transfer(left, right):
        a, n = position(old, left)
        b, m = position(new, right)
        if n != m or np.any(source_class[a:a+n]) or np.any(destination_covered[b:b+n]):
            raise ValueError('Source or destination words are aliased')
        output[b:b+n] = source[a:a+n]
        source_class[a:a+n] = 1
        destination_covered[b:b+n] = True
        transfers.append((a, b, n))

    for prefix in plan['bank_prefixes']:
        if prefix['bytes']:
            address = dict(pe=prefix['pe'], byte_offset=0, bytes=prefix['bytes'])
            transfer(address, address)
    dropped = 0
    for obj in plan['objects']:
        start = obj['page_start']
        for page in range(start, start+obj['retained_pages']):
            transfer(src.owner(page), dst.owner(page))
        for page in range(start+obj['retained_pages'], start+obj['source_pages']):
            if obj['kind'] != 'request_state':
                raise ValueError('Cannot discard original weights or temporary work buffers')
            a, n = position(old, src.owner(page))
            if np.any(source_class[a:a+n]) or np.any(source[a:a+n]):
                raise ValueError('Discarded source state is live or overlaps retained data')
            source_class[a:a+n] = 2
            dropped += n
    if (not np.all(source_class) or not np.all(destination_covered)
            or np.count_nonzero(source_class == 1) != cursor
            or dropped*4 != plan['reclaimed_bytes']):
        raise ValueError('Incomplete retained/removed original word partition')
    del source_class, destination_covered
    output.flush()
    del output
    actual = np.load('banks.npy', mmap_mode='r')
    verified = 0
    for a, b, n in transfers:
        if not np.array_equal(source[a:a+n], actual[b:b+n]):
            raise ValueError('Retained original word changed on fresh readback')
        verified += n
    if verified != cursor:
        raise ValueError('Fresh readback omitted a candidate word')
    index = dict(records=list(new.values()), total_words=cursor, allocated_bytes=cursor*4,
                 initialized_weight_and_scale_bytes=oldindex['initialized_weight_and_scale_bytes'],
                 remaining_bytes_are_zero_state_or_values=True, conversations=1, work_slots=2,
                 bank_specialization='dialogue-bank-placement.json')
    Path('bank-index.json').write_text(json.dumps(index, separators=(',', ':'))+'\n')
    result = dict(passed=True, physical=False, neural_execution=False, multi_turn_qualified=False,
                  all_source_words_classified_once=True, all_destination_words_verified_once=True,
                  retained_words_verified=verified, discarded_zero_state_words=dropped,
                  allocated_bytes=cursor*4, reclaimed_bytes=dropped*4, transfers=len(transfers),
                  original_weight_and_scale_bytes=index['initialized_weight_and_scale_bytes'],
                  every_original_matrix_and_mlp_prefix_preserved=True,
                  retained_auxiliary_pages=plan['retained_pages'], work_slots=2, conversations=1,
                  source_hashes=config['hashes'],
                  hashes={name: sha(Path(name)) for name in ['banks.npy', 'bank-index.json']},
                  seconds=time.monotonic()-started)
    Path('bank-remap-proof.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
