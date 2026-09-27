"""Four compact reduction forests for all aligned original projection banks.

The path embedding includes skipped BF16 state banks and actor-row bridge PEs.
Abstract six-bit words encode one RX and one TX port; no SDK register encoding or
runtime route transition is claimed here. Dense arrays are generated remotely.
"""
from .contraction import tree

PORTS = dict(NONE=0, RAMP=1, NORTH=2, SOUTH=3, WEST=4, EAST=5)
COLORS = tuple(range(3, 16))


def pattern(k):
    if k not in (40, 48, 136):
        raise ValueError('Original complete K widths required')
    vertices = [dict(active=0, source=0, destination=0) for _ in range(k)]
    cuts = [0] * (k + 1)
    for node in tree(k)[1:]:
        child, parent = node['rank'], node['parent']
        index = 2 * (node['depth'] - 1) + node['side']; bit = 1 << index
        if index >= len(COLORS):
            raise ValueError('Color domain')
        for rank in range(parent, child + 1):
            if vertices[rank]['active'] & bit:
                raise ValueError('Two distinct edges share a PE/color')
            vertices[rank]['active'] |= bit
        vertices[child]['source'] |= bit; vertices[parent]['destination'] |= bit
        for cut in range(parent + 1, child + 1):
            if cuts[cut] & bit:
                raise ValueError('Edge-cut collision')
            cuts[cut] |= bit
    return dict(k_blocks=k, vertices=vertices, cuts=cuts, colors=list(COLORS),
                numerical_order='P9 native local dots followed by fixed preorder FP32 tree; complete original serial-block bit equivalence is not assumed.')


def path_embedding(atlas):
    import numpy as np
    p = atlas['policy']; width = p['width']; count = p['bank_pes']; rows = np.asarray(atlas['geometry']['bank_rows'], dtype=np.int32)
    ids = np.arange(count, dtype=np.int32); row = ids // width; column = ids % width
    gaps = np.r_[0, np.cumsum(np.diff(rows) - 1)]
    positions = ids + gaps[row]
    xy = np.full((int(positions[-1]) + 1, 2), -1, dtype=np.int32)
    xy[positions, 0] = np.where(row % 2 == 0, column, width - 1 - column); xy[positions, 1] = rows[row]
    for r in range(int(row[-1])):
        before = (r + 1) * width - 1; x = int(xy[positions[before], 0])
        for offset, y in enumerate(range(int(rows[r]) + 1, int(rows[r + 1]))):
            xy[positions[before] + 1 + offset] = [x, y]
    if np.any(xy < 0) or not np.all(np.abs(np.diff(xy, axis=0)).sum(axis=1) == 1):
        raise ValueError('Incomplete or non-neighbor path embedding')
    previous = np.zeros(len(xy), dtype=np.uint16); following = np.zeros_like(previous)
    delta = np.diff(xy, axis=0)
    for difference, name, opposite in [((1, 0), 'EAST', 'WEST'), ((-1, 0), 'WEST', 'EAST'),
                                      ((0, 1), 'SOUTH', 'NORTH'), ((0, -1), 'NORTH', 'SOUTH')]:
        at = np.flatnonzero(np.all(delta == difference, axis=1))
        following[at] = PORTS[name]; previous[at + 1] = PORTS[opposite]
    return xy, positions.astype(np.int32), previous, following


def eligible_banks(atlas):
    import numpy as np
    segments = atlas['bf16_eligible_segments']
    return np.concatenate([np.arange(s['bank_start'], s['bank_start'] + s['count'], dtype=np.int32) for s in segments])


def lower(atlas, kind, k, embedding=None):
    import numpy as np
    if kind not in ('fp8', 'bf16') or kind == 'bf16' and k != 40:
        raise ValueError('Forest profile')
    xy, bank_positions, previous, following = path_embedding(atlas) if embedding is None else embedding
    selected = np.arange(len(bank_positions), dtype=np.int32) if kind == 'fp8' else eligible_banks(atlas)
    nodes = bank_positions[selected]
    if len(nodes) % k:
        raise ValueError('Partial row group')
    model = pattern(k); rank_after = np.searchsorted(nodes, np.arange(len(xy)), side='left')
    bounded = np.minimum(rank_after, len(nodes) - 1); is_node = (rank_after < len(nodes)) & (nodes[bounded] == np.arange(len(xy)))
    rank = rank_after % k
    vertex_active = np.array([v['active'] for v in model['vertices']], np.uint16)
    vertex_source = np.array([v['source'] for v in model['vertices']], np.uint16)
    vertex_destination = np.array([v['destination'] for v in model['vertices']], np.uint16)
    cut = np.asarray(model['cuts'], np.uint16)
    active = np.where(is_node, vertex_active[rank], cut[rank])
    source = np.where(is_node, vertex_source[rank], 0); destination = np.where(is_node, vertex_destination[rank], 0)
    words = np.zeros((len(xy), len(COLORS)), np.uint16)
    for index in range(len(COLORS)):
        bit = 1 << index; enabled = (active & bit) != 0
        rx = np.where((source & bit) != 0, PORTS['RAMP'], following)
        tx = np.where((destination & bit) != 0, PORTS['RAMP'], previous)
        if np.any(enabled & ((rx == 0) | (tx == 0))):
            raise ValueError('Route leaves path')
        words[:, index] = np.where(enabled, rx | (tx << 3), 0)
    return words, nodes


def document(atlas, bundles):
    if bundles['provenance']['atlas_sha256'] == '':
        raise ValueError('Atlas provenance required')
    profiles = [dict(name=kind + '-k' + str(k), kind=kind, **pattern(k)) for kind, k in
                [('fp8', 40), ('fp8', 48), ('fp8', 136), ('bf16', 40)]]
    names = {p['name'] for p in profiles}
    for bundle in bundles['bundles']:
        if bundle['profile'] not in names:
            raise ValueError('Missing complete projection profile')
        c = atlas['classes'][bundle['kind']]; k = bundle['k_blocks']
        if c['pes'] % k or c['phase'] % k or bundle['stream_start'] % k:
            raise ValueError('Matrix changes row grouping')
        for segment in bundle['segments']:
            if segment['class_start'] % k or segment['count'] % k:
                raise ValueError('Segment splits a group')
    return dict(schema='wse-complete-reduction-forests-v1', atlas_sha256=bundles['provenance']['atlas_sha256'],
                profiles=profiles, ports=PORTS, colors=list(COLORS),
                abstract_word='rx_port_code | (tx_port_code << 3); zero disables route; not an SDK register image',
                bank_path='P12 serpentine bank path plus every actor-row bridge coordinate; BF16-excluded banks forward without contributing.',
                applicability='Every aligned whole-K matrix segment activates complete disjoint trees in one of four profiles. Multiple resident slots reuse the same forest sequentially.',
                input_multicast_admitted=False, output_scatter_admitted=False, route_epochs_admitted=False,
                compiled_sram_admitted=False, full_model_executable=False, full_model_speed_target_achieved=False)
