"""Complete bank overlay and paired-packet column multicast for original FP8.

Pairing row residues reduces column channels to 2/4/1 for K40/K48/K136.
Only K136 moves into a 748-column subring; tensor contents and all actor/state
coordinates stay fixed. Filter windows are abstract until a backend qualifies
the installed SDK's half-wavelet counter semantics.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def segments(start, count, ring, k):
    out = []; offset = 0
    while offset < count:
        address = start + offset; at = address % ring; size = min(count - offset, ring - at)
        if at % k or size % k:
            raise ValueError('Partial K group')
        out.append(dict(class_start=at, count=size, slot=address // ring,
                        tile_start=offset, row_start=offset // k))
        offset += size
    return out


def pairs(k):
    if k in (40, 48):
        return [(i, i + k // 2) for i in range(k // 2)]
    if k == 136:
        return [(i, 135 - i) for i in range(68)]
    raise ValueError('Original FP8 K width required')


def packet_groups(k):
    pair_list = pairs(k); per_row = 6 if k == 136 else 2
    return [dict(actor_row=i // per_row, pair_start=i, pairs=pair_list[i:i + per_row],
                 packet_blocks=[v for pair in pair_list[i:i + per_row] for v in pair])
            for i in range(0, len(pair_list), per_row)]


def source(k, x, lane):
    lanes = {40: 2, 48: 4, 136: 1}[k]
    if not 0 <= lane < lanes or not 0 <= x < 750 or k == 136 and not 1 <= x <= 748:
        raise ValueError('Input source coordinate/lane')
    if k == 136:
        a = (x - 1) % 136; pair_id = min(a, 135 - a); per_row = 6
    else:
        owner = lane * 750 + (x if lane % 2 == 0 else 749 - x)
        pair_id = owner % (k // 2); per_row = 2
    group = packet_groups(k)[pair_id // per_row]
    return dict(column=x, lane=lane, actor_row=pair_id // per_row,
                pair_id=pair_id, blocks=list(pairs(k)[pair_id]),
                horizontal_window=dict(offset_words=(pair_id % per_row) * 130,
                                       pass_words=130, period_words=len(group['packet_blocks']) * 65),
                vertical_color=21 - lanes + lane, retained_packet_words=130)


def consumer(k, row, x):
    """row is P12 logical bank-row index, excluding actor rows."""
    if not 0 <= row < 1148 or not 0 <= x < 750:
        raise ValueError('Bank coordinate')
    bank = row * 750 + (x if row % 2 == 0 else 749 - x)
    if bank >= 860880 or k == 136 and (row >= 1146 or not 1 <= x <= 748):
        return None
    lanes = {40: 2, 48: 4, 136: 1}[k]; lane = row % lanes
    rank = bank if k != 136 else row * 748 + (x - 1 if row % 2 == 0 else 748 - x)
    block = rank % k; src = source(k, x, lane)
    if block not in src['blocks']:
        raise ValueError('Source pairing misses actual tensor K block')
    ordinal = src['blocks'].index(block)
    return dict(lane=lane, block=block, source_actor_row=src['actor_row'], vertical_color=src['vertical_color'],
                vertical_window=dict(offset_words=ordinal * 65, pass_words=65, period_words=130))


def build(atlas_path):
    path = Path(atlas_path); atlas = json.loads(path.read_text())
    if atlas['policy']['bank_pes'] != 860880 or len(atlas['geometry']['bank_rows']) != 1148:
        raise ValueError('Pinned P12 geometry required')
    classes = {
        'fp8-main': dict(kind='fp8', pes=860880, phase=0, tile_bytes=260,
                         geometry='Original P12 bank IDs', k_blocks=[40, 48]),
        'fp8-wide': dict(kind='fp8', pes=857208, phase=0, tile_bytes=260,
                         geometry='First1146 P12 bank rows, physical columns1..748, serpentine', k_blocks=[136]),
        'bf16': dict(kind='bf16', pes=824000, phase=480000, tile_bytes=512,
                     geometry='Unchanged P12 eligible bank ring; new aligned phase', k_blocks=[40])}
    for c in classes.values():
        c.update(stream_span=0, real_tiles=0, padding_tiles=0)
    matrices = []
    for original in atlas['matrices']:
        m = deepcopy(original); key = 'bf16' if m['kind'] == 'bf16' else 'fp8-wide' if m['k_blocks'] == 136 else 'fp8-main'
        c = classes[key]; k = m['k_blocks']; start = (c['stream_span'] + k - 1) // k * k
        if c['pes'] % k or c['phase'] % k:
            raise ValueError('Class alignment')
        m.update(storage_class=key, stream_start=start, padding_before=start - c['stream_span'])
        m['segments'] = segments(start, m['tiles'], c['pes'], k)
        c['real_tiles'] += m['tiles']; c['padding_tiles'] += m['padding_before']; c['stream_span'] = start + m['tiles']
        matrices.append(m)
    profiles = []
    for k, lanes, tree_colors in [(40, 2, 9), (48, 4, 10), (136, 1, 13)]:
        cols = range(1, 749) if k == 136 else range(750)
        profiles.append(dict(k_blocks=k, horizontal_color=2, vertical_colors=list(range(21 - lanes, 21)),
                             reduction_colors=list(range(3, 3 + tree_colors)),
                             unused_application_colors=sorted(set(range(2, 21)) - {2} - set(range(21 - lanes, 21)) - set(range(3, 3 + tree_colors))),
                             horizontal_groups=packet_groups(k),
                             sources=[source(k, x, lane) for x in cols for lane in range(lanes)],
                             vertical_packet_words=130, bank_packet_words=65,
                             max_horizontal_stream_words=max(len(g['packet_blocks']) * 65 for g in packet_groups(k)),
                             max_column_stream_words=lanes * 130,
                             source_buffer_bytes=520, bank_input_buffer_bytes=260,
                             geometry_note='Horizontal actor-row streams select pairs; each column source broadcasts its pair both north and south. Bank counter filters select one complete packet.'))
    preserved = ['actors', 'gdn', 'attention', 'canonical_tensors', 'canonical_pages', 'value_arena',
                 'normalizations', 'quant_actor_ids', 'quant_storage', 'scale_aliases', 'bf16_eligible_segments']
    hashes = {n: hashlib.sha256(json.dumps(atlas[n], sort_keys=True, separators=(',', ':')).encode()).hexdigest() for n in preserved}
    return dict(schema='wse-paired-column-input-overlay-v1', model=atlas['model'], revision=atlas['revision'],
                base_atlas_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), preserved_base_fields_sha256=hashes,
                classes=classes, matrices=matrices, input_profiles=profiles,
                address='owner=(phase+stream_start+tile)%class.pes; slot=(stream_start+tile)//class.pes. FP8-wide local storage begins after this PE\'s complete FP8-main slot reservation.',
                metrics=dict(original_matrices=len(matrices), original_real_tiles=sum(m['tiles'] for m in matrices),
                             resident_matrix_bytes=sum(c['stream_span'] * c['tile_bytes'] for c in classes.values())),
                storage_audited=False, input_routes_audited=False, counter_filter_backend_qualified=False,
                complete_input_executable=False, compiled_sram_admitted=False, full_model_executable=False,
                full_model_speed_target_achieved=False,
                unresolved=['Original quant/value owners to horizontal stream producers',
                            'BF16 operand distribution', 'Complete result scatter to value arena',
                            'Whole graph route epoch readiness and combined neural SRAM',
                            'Complete physical model numerical and speed acceptance'])


def lower_input_routes(atlas, profile):
    """Remote dense abstract RX + TX-bitmask planes, not SDK register encodings."""
    import numpy as np
    height, width = 1160, 750; k = profile['k_blocks']; lanes = len(profile['vertical_colors'])
    bank_rows = np.asarray(atlas['geometry']['bank_rows']); actor_rows = atlas['geometry']['actor_rows']
    rows = np.full(height, -1, np.int32); rows[bank_rows] = np.arange(len(bank_rows))
    yy, xx = np.indices((height, width)); rr = rows[:, None]
    bank_id = rr * width + np.where(rr % 2 == 0, xx, width - 1 - xx)
    available = (rr >= 0) & (bank_id < 860880)
    if k == 136:
        available &= (rr < 1146) & (xx >= 1) & (xx <= 748)
    words = np.zeros((lanes + 1, height, width), np.uint16)
    ingress = np.zeros((height, width), bool)
    for record in profile['sources']:
        ingress[actor_rows[record['actor_row']], record['column']] = True
    # TX bits: RAMP0, NORTH1, SOUTH2, WEST3, EAST4. RX uses P14 port codes.
    for group in profile['horizontal_groups']:
        y = actor_rows[group['actor_row']]
        tx = (np.arange(width) < width - 1).astype(np.uint16) * 16 + ingress[y]
        rx = np.full(width, 4, np.uint16); rx[0] = 1
        words[0, y] = rx | (tx << 3)
    for lane in range(lanes):
        root = np.full(width, -1, np.int32)
        for record in profile['sources']:
            if record['lane'] == lane:
                root[record['column']] = actor_rows[record['actor_row']]
        recipients = available & (rr % lanes == lane)
        top = np.where(recipients, yy, height).min(axis=0)
        bottom = np.where(recipients, yy, -1).max(axis=0)
        enabled = (root[None, :] >= 0) & (yy >= top[None, :]) & (yy <= bottom[None, :])
        rx = np.where(yy == root[None, :], 1, np.where(yy < root[None, :], 3, 2))
        north = (yy <= root[None, :]) & (yy > top[None, :])
        south = (yy >= root[None, :]) & (yy < bottom[None, :])
        tx = recipients.astype(np.uint16) + north.astype(np.uint16) * 2 + south.astype(np.uint16) * 4
        words[lane + 1] = np.where(enabled, rx | (tx << 3), 0)
    return words
