"""Validate complete original projection coverage before any hardware allocation."""
SHAPES = ((10240, 5120), (6144, 5120), (48, 5120), (48, 5120), (5120, 6144))


def validate(stage, network, setups, profiles, index):
    origin = stage['rect'][:2]
    if stage['id'] != 'layer_00' or stage['rect'] != [63, 0, 78, 146]:
        raise ValueError('Original complete stage geometry')
    controller = stage['request_controller']['pe']
    workers = network['workers']
    if network['gateway'] != controller or len(workers) != 3554:
        raise ValueError('Complete routed mixer required')
    by_pe = {tuple(w['pe']): w['tag'] for w in workers}
    if len(by_pe) != 3554 or set(by_pe.values()) != set(range(3554)):
        raise ValueError('Missing or duplicate internal target')
    descriptors = {tuple(pe): list(s) for pe, s in setups}
    if len(descriptors) != len(setups) or set(descriptors) != set(by_pe):
        raise ValueError('Complete descriptor ownership')
    capacities = {tuple(pe): p['parameters'].get('bank_words', 0)
                  for p in profiles['profiles'] for pe in p['pes']}
    records = index['records']
    if len({tuple(r['pe']) for r in records}) != len(records):
        raise ValueError('Duplicate original bank record')
    cursor = 0
    for r in records:
        if r['offset'] != cursor or r['words'] != capacities[tuple(r['pe'])] or r['words'] <= 0:
            raise ValueError('Original bank extent/offset')
        cursor += r['words']
    if cursor != index['total_words'] or 4 * cursor != index['allocated_bytes'] or 4 * cursor != 396953216:
        raise ValueError('Complete original bank byte extent')
    if {tuple(r['pe']) for r in records} != {pe for pe, n in capacities.items() if n}:
        raise ValueError('Missing resident bank')
    roots, iterations = [], []
    by_tag = {tag: descriptors[pe] for pe, tag in by_pe.items()}
    for mi, (rows, columns) in enumerate(SHAPES):
        parts, width = columns // 128, 1 if mi in (2, 3) else 2
        tiles = set()
        current = []
        counts = {}
        for pe, tag in by_pe.items():
            s = descriptors[pe]
            if len(s) != 40 or any(type(v) is not int or not 0 <= v < 65536 for v in s):
                raise ValueError('Descriptor ABI')
            base, count, first, stride, key, keys, bf16, nrows = s[mi * 8:mi * 8 + 8]
            if (key, keys, bf16, nrows) != (tag % parts, parts, int(width == 1), width):
                raise ValueError('Full K family/precision/row extent')
            if count and base + count * (64 if bf16 else 65) > capacities[pe]:
                raise ValueError('Weight access exceeds compiled bank')
            counts[tag] = count
            for i in range(count):
                row = first + stride * i
                if row % width or row + width > rows or (row, key) in tiles:
                    raise ValueError('Missing/overlapping original matrix tile')
                tiles.add((row, key))
            if count and key == 0:
                for k in range(parts):
                    peer = by_tag[tag + k][mi * 8:mi * 8 + 8]
                    if peer[1:4] != [count, first, stride]:
                        raise ValueError('Full K peers disagree on root row sequence')
                current.append(dict(tag=tag, pe=list(pe), count=count, first=first, stride=stride, rows=width))
        if tiles != {(r, k) for r in range(0, rows, width) for k in range(parts)}:
            raise ValueError('Incomplete original matrix')
        roots.append(sorted(current, key=lambda r: r['tag']))
        iterations.append(counts)
    return dict(origin=origin, gateway=[controller[0]-origin[0], controller[1]-origin[1]],
                targets=by_pe, roots=roots, iterations=iterations, descriptors=descriptors,
                sdk_banks=[r for r in records if tuple(r['pe']) not in by_pe],
                device_banks=[r for r in records if tuple(r['pe']) in by_pe])


def root_rows(roots):
    """Interleave independent full-K roots; preserve each root's exact row order."""
    for i in range(max(r['count'] for r in roots)):
        for r in roots:
            if i < r['count']:
                yield r, r['first'] + i * r['stride']


def canonical_bf16(values):
    """Only the predeclared sign of zero may differ at the final boundary."""
    import numpy as np
    values = np.asarray(values, dtype=np.uint16)
    return np.where((values & 0x7fff) == 0, 0, values).astype(np.uint16)
