"""Independent exhaustive metadata audit; no weight payloads or hardware use.

Does not import the atlas builder. NumPy visits every real matrix tile in bounded
chunks, cross-checking compact dispatch descriptors against tensor coordinates.
The only large arrays are address-occupancy maps (<220 MiB combined).
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import numpy as np


def require(test, message):
    if not test:
        raise ValueError(message)


def audit(atlas_path, config_dir, chunk=131072):
    started = time.monotonic()
    atlas = json.loads(atlas_path.read_text())
    graph = json.loads((config_dir / 'model-graph.json').read_text())
    specs = json.loads((config_dir / 'tensors.json').read_text())['tensors']
    for name, digest in atlas['provenance'].items():
        require(hashlib.sha256((config_dir / name).read_bytes()).hexdigest() == digest,
                'Changed source metadata: ' + name)
    require(atlas['model'] == graph['model'] and atlas['revision'] == graph['revision'], 'Identity')
    require(not any(atlas[k] for k in ['physical_routes_admitted', 'compiled_sram_admitted',
                                     'full_model_executable', 'full_model_speed_target_achieved']), 'Scope')
    p = atlas['policy']
    width, height, banks = p['width'], p['height'], p['bank_pes']
    require((width, height, banks) == (750, 1160, 860880), 'Pinned geometry')
    actor_rows = set(atlas['geometry']['actor_rows'])
    bank_xy = []
    actors_xy = []
    for y in range(height):
        if y in actor_rows:
            actors_xy.extend((x, y) for x in range(width))
        else:
            row = sum(yy not in actor_rows for yy in range(y))
            xs = range(width) if row % 2 == 0 else range(width - 1, -1, -1)
            for x in xs:
                if len(bank_xy) < banks:
                    bank_xy.append((x, y))
                else:
                    actors_xy.append((x, y))
    actors_xy.sort(key=lambda xy: (xy[1], xy[0]))
    require(len(bank_xy) + len(actors_xy) == width * height, 'Complete fabric')
    require([tuple(a['xy']) for a in atlas['actors']] == actors_xy, 'Actor geometry')
    require([a['id'] for a in atlas['actors']] == list(range(len(actors_xy))), 'Actor IDs')
    bank_id = {xy: b for b, xy in enumerate(bank_xy)}
    require(len(bank_id) == banks and not set(bank_id).intersection(actors_xy), 'Geometry bijection')
    require(atlas['geometry']['bank_rows'] == [y for y in range(height) if y not in actor_rows], 'Bank rows')
    roles = {}

    def claim(a, role):
        require(isinstance(a, int) and 0 <= a < len(actors_xy) and a not in roles, 'Actor alias')
        roles[a] = role

    state_banks = set()
    state_keys = set()
    expected_gdn = {int(n['attributes']['state'].split('.')[1]) for n in graph['nodes']
                    if n['op'] == 'gated_delta_recurrence'}
    for g in atlas['gdn']:
        key = (g['layer'], g['value_head'])
        require(key not in state_keys and key[0] in expected_gdn and 0 <= key[1] < 48, 'GDN key')
        state_keys.add(key)
        require(g['key_head'] == key[1] // 3 and g['state_shape_per_pe'] == [32, 32]
                and g['state_bytes_per_pe'] == 4096, 'GDN dimensions')
        claim(g['controller'], 'gdn_controller')
        x, y, nx, ny = g['state_rectangle']
        require((nx, ny) == (4, 4), 'GDN shards')
        coverage = np.zeros((128, 128), dtype=np.uint8)
        for vs in range(4):
            for ks in range(4):
                b = bank_id.get((x + ks, y + vs))
                require(b is not None and b not in state_banks, 'State bank alias')
                state_banks.add(b)
                coverage[ks * 32:(ks + 1) * 32, vs * 32:(vs + 1) * 32] += 1
        require(bool(np.all(coverage == 1)), 'GDN complete key/value axes')
        require(g['conv_channel_starts'] == [key[1] // 3 * 128, 2048 + key[1] // 3 * 128,
                                             4096 + key[1] * 128], 'Convolution source')
        require(g['history_bytes'] == 3072 and g['parameter_bytes'] == 3332, 'GDN controller bytes')
    require(state_keys == {(l, h) for l in expected_gdn for h in range(48)}, 'Full GDN coverage')
    semantic_state_count = len(state_banks)
    x, y, nx, ny = atlas['state_class_spare_rectangle']
    for yy in range(y, y + ny):
        for xx in range(x, x + nx):
            b = bank_id.get((xx, yy))
            require(b is not None and b not in state_banks, 'Spare state alias')
            state_banks.add(b)
    eligible = np.array([b for b in range(banks) if b not in state_banks], dtype=np.int64)
    compact = 0
    for s in atlas['bf16_eligible_segments']:
        n = s['count']
        require(s['compact_start'] == compact and n > 0, 'Eligible continuity')
        require(np.array_equal(eligible[compact:compact + n], np.arange(s['bank_start'], s['bank_start'] + n)),
                'Eligible segment mismatch')
        compact += n
    require(compact == len(eligible) == atlas['classes']['bf16']['pes'], 'Complete eligible map')

    expected_attention = {int(n['attributes']['state'].split('.')[1]) for n in graph['nodes']
                          if n['op'] == 'kv_append'}
    attention_keys = set()
    kv_bytes = 0
    for a in atlas['attention']:
        key = (a['layer'], a['kv_head'])
        require(key not in attention_keys and key[0] in expected_attention and 0 <= key[1] < 4, 'Attention key')
        attention_keys.add(key)
        claim(a['controller'], 'attention_controller')
        require(a['query_heads'] == list(range(key[1] * 6, key[1] * 6 + 6)), 'GQA mapping')
        require(a['parameter_bytes'] == 1152, 'Attention controller bytes')
        coverage = np.zeros((96, 256), dtype=np.uint8)
        for s in a['shards']:
            claim(s['actor'], 'attention_kv')
            t, nt, d, nd = (s[k] for k in ['token_start', 'tokens', 'channel_start', 'channels'])
            require(nt > 0 and nd > 0 and 0 <= t < t + nt <= 96 and 0 <= d < d + nd <= 256, 'KV bounds')
            require(s['kv_bytes'] == nt * nd * 4 == 3072, 'KV BF16 K and V bytes')
            coverage[t:t + nt, d:d + nd] += 1
            kv_bytes += s['kv_bytes']
        require(np.all(coverage == 1), 'KV complete nonoverlapping coverage')
    require(attention_keys == {(l, h) for l in expected_attention for h in range(4)}, 'Full attention coverage')

    page_intervals = {}
    for page in atlas['canonical_pages']:
        claim(page['actor'], 'canonical_weights')
        require(page['capacity_bytes'] == 32768, 'Canonical page capacity')
        page_intervals[page['actor']] = []
    for name, t in atlas['canonical_tensors'].items():
        spec = specs[name]
        require(t['dtype'] == spec['dtype'] == 'BF16' and t['shape'] == spec['shape']
                and t['bytes'] == spec['bytes'], 'Canonical tensor identity')
        offset = 0
        for c in t['chunks']:
            a, pos, size = c['actor'], c['actor_byte_offset'], c['bytes']
            require(a in page_intervals and c['tensor_byte_offset'] == offset and size > 0
                    and 0 <= pos < pos + size <= 32768, 'Canonical chunk bounds')
            page_intervals[a].append((pos, pos + size))
            offset += size
        require(offset == spec['bytes'], 'Complete canonical bytes')
    for iv in page_intervals.values():
        iv.sort()
        require(all(iv[i][1] <= iv[i + 1][0] for i in range(len(iv) - 1)), 'Canonical page alias')
    claim(atlas['global_controller'], 'global_control')
    for a in atlas['quant_actor_ids']:
        claim(a, 'quant_value')
    require(len(atlas['quant_actor_ids']) == 4352, 'Quant producer count')
    require(all(a['role'] == roles.get(a['id'], 'spare') for a in atlas['actors']), 'Actor role assignment')
    require(dict(Counter(a['role'] for a in atlas['actors'])) == atlas['metrics']['actor_roles'], 'Role census')

    names = list(dict.fromkeys(w for n in graph['nodes'] for w in n['weights']))
    bound = set(atlas['canonical_tensors']) | set(atlas['scale_aliases'])
    matrices = atlas['matrices']
    require(len(matrices) == 498 and [m['id'] for m in matrices] == list(range(498)), 'Matrix IDs')
    visited = Counter()
    bank_counts = {}
    for kind, c in atlas['classes'].items():
        n, phase = c['pes'], c['phase']
        slots = (c['stream_span'] + n - 1) // n
        occupancy = np.zeros((slots, n), dtype=np.uint16)
        owner_counts = np.zeros(n, dtype=np.int64)
        end = padding = 0
        for m in (m for m in matrices if m['kind'] == kind):
            name = m['tensor']; spec = specs[name]
            rows, columns = spec['shape']; k = columns // 128
            require(name not in bound and m['shape'] == [rows, columns] and rows % 2 == 0
                    and columns % 128 == 0 and m['row_tiles'] == rows // 2 and m['k_blocks'] == k, 'Matrix shape/alias')
            bound.add(name)
            start = ((end + k - 1) // k) * k
            size = (rows // 2) * k
            require(m['stream_start'] == start and m['tiles'] == size and m['padding_before'] == start - end,
                    'Independent canonical stream prefix')
            require(n % k == phase % k == 0, 'Whole K ring')
            require(m['mode'] == ('lookup' if name.endswith('embed_tokens.weight') else 'gemv'), 'Embedding mode')
            require(spec['dtype'] == ('F8_E4M3' if kind == 'fp8' else 'BF16'), 'Matrix dtype')
            padding += start - end; end = start + size
            if kind == 'fp8':
                alias = atlas['scale_aliases'][name + '_scale_inv']
                require(alias['matrix'] == m['id'] and specs[name + '_scale_inv']['shape'] == [(rows + 127) // 128, k], 'Original scales')
            relative = 0
            for s in m['segments']:
                require(s['tile_start'] == relative and s['row_start'] == relative // k
                        and s['count'] > 0 and s['count'] % k == 0 and s['class_start'] % k == 0,
                        'Dispatch whole-row prefix')
                require(0 <= s['class_start'] < s['class_start'] + s['count'] <= n, 'Dispatch span')
                for offset in range(0, s['count'], chunk):
                    count = min(chunk, s['count'] - offset)
                    # Dispatch reconstruction has no per-tile table or builder call.
                    normalized = np.arange(s['class_start'] + offset, s['class_start'] + offset + count, dtype=np.int64)
                    owner = (normalized + phase) % n
                    decoded_tensor_tile = s['slot'] * n + normalized - start
                    expected_tile = np.arange(relative + offset, relative + offset + count, dtype=np.int64)
                    require(np.array_equal(decoded_tensor_tile, expected_tile), 'Dispatch/tensor address disagreement')
                    require(np.all((decoded_tensor_tile // k) * 2 + 1 < rows)
                            and np.all((decoded_tensor_tile % k + 1) * 128 <= columns), 'Original tensor tile bounds')
                    require(not np.any(occupancy[s['slot'], owner]), 'Duplicate physical resident address')
                    occupancy[s['slot'], owner] = m['id'] + 1
                    owner_counts[owner] += 1
                    visited[kind] += count
                relative += s['count']
            require(relative == size, 'Complete matrix descriptor coverage')
        require(end == c['stream_span'] and padding == c['padding_tiles'] and visited[kind] == c['real_tiles'], 'Class conservation')
        require(int(np.count_nonzero(occupancy)) == c['real_tiles'], 'All unique addresses')
        # Padding is retained as explicit zero slots; total reserved slots per
        # owner is independently enumerated from full stream span, not real count.
        q, rem = divmod(end, n)
        reserved = np.full(n, q, dtype=np.int64)
        reserved[(np.arange(rem) + phase) % n] += 1
        require(np.all(owner_counts <= reserved) and int((reserved - owner_counts).sum()) == padding, 'Padding occupancy')
        physical = np.zeros(banks, dtype=np.int64)
        physical[np.arange(banks) if kind == 'fp8' else eligible] = reserved
        bank_counts[kind] = physical
        del occupancy
    require(bound == set(names) and len(bound) == 1251, 'All original tensor binding')
    states = np.zeros(banks, dtype=np.int64)
    states[list(state_banks)] = 4096
    census = Counter(zip(bank_counts['fp8'].tolist(), bank_counts['bf16'].tolist(), states.tolist()))
    expected_census = Counter({(r['fp8_slots'], r['bf16_slots'], r['state_capacity_bytes']): r['pes'] for r in atlas['bank_profiles']})
    require(census == expected_census, 'Independently enumerated SRAM data census')
    for r in atlas['bank_profiles']:
        data = r['fp8_slots'] * 260 + r['bf16_slots'] * 512 + r['state_capacity_bytes']
        require(data == r['data_bytes'] and r['remaining_code_stack_scratch_bytes'] == 48128 - data > 0, 'Bank payload budget')

    # Pairwise checking is independent of the builder's free-list allocator.
    arena = atlas['value_arena']; values = arena['values']; ends = {}; births = {}
    for node in graph['nodes']:
        for v in node['outputs']:
            births[v] = ends[v] = node['id']
        for v in node['inputs']:
            if v != 'token':
                ends[v] = node['id']
    ends[graph['selected_token']] = len(graph['nodes'])
    require(set(values) == set(births), 'All graph values')
    records = list(values.values())
    for name, r in values.items():
        spec = graph['values'][name]; cells = (math.prod(spec['shape']) + 3) // 4
        require(r['shape'] == spec['shape'] and r['dtype'] == spec['dtype'] and r['cells'] == cells
                and r['elements'] == math.prod(spec['shape']) and r['base_cell'] % 32 == 0
                and r['reserved_cells'] == (cells + 31) // 32 * 32
                and r['birth'] == births[name] and r['last_use'] == ends[name], 'Value extent/lifetime')
        last_cell = r['base_cell'] + cells - 1
        require(last_cell // 4352 < arena['local_cells_per_actor'], 'Striped local value bound')
    overlap_pairs = 0
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if max(a['birth'], b['birth']) <= min(a['last_use'], b['last_use']):
                overlap_pairs += 1
                require(a['base_cell'] + a['reserved_cells'] <= b['base_cell']
                        or b['base_cell'] + b['reserved_cells'] <= a['base_cell'], 'Live value alias')
    high = max(r['base_cell'] + r['reserved_cells'] for r in records)
    peak = max(sum(r['reserved_cells'] for r in records if r['birth'] <= n <= r['last_use'])
               for n in range(len(graph['nodes']) + 1))
    require((high, peak) == (arena['high_water_cells'], arena['peak_live_reserved_cells']), 'Arena highwater/peak')
    require(arena['bytes_per_actor'] == math.ceil(high / 4352) * 16, 'Value SRAM')
    norm_nodes = [n for n in graph['nodes'] if n['op'] == 'zero_centered_rmsnorm']
    require(len(norm_nodes) == len(atlas['normalizations']) == 129, 'Norm completeness')
    for i, (node, n) in enumerate(zip(norm_nodes, atlas['normalizations'])):
        require(n['node'] == node['id'] and n['input'] == node['inputs'][0] and n['weight'] == node['weights'][0]
                and n['bank'] == i and n['owners'] == 1280 and n['elements_per_owner'] == 4
                and n['owner_start'] == values[n['input']]['base_cell'] % 4352, 'Norm gain placement')
    q = atlas['quant_storage']
    capture = 96 * (64 * 5120 + 5120 + 248320) * 2
    require(q['capture_required_bytes'] == capture <= q['capture_capacity_bytes_per_pe'] * 4352, 'Capture coverage')
    require(q['remaining_code_stack_other_scratch_bytes'] == 48128 - arena['bytes_per_actor'] - 129 * 8 - 28672, 'Quant SRAM reservation')
    return dict(schema='wse-complete-model-atlas-audit-v1', passed=True, physical=False,
                weights_read=False, full_model_executable=False, full_model_speed_target_achieved=False,
                atlas_sha256=hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_metadata_sha256=atlas['provenance'], matrix_tiles_visited=dict(visited),
                original_tensors_checked=len(bound), application_pes=width * height,
                actor_pes=len(actors_xy), bank_pes=banks, gdn_state_pes=semantic_state_count,
                kv_cache_bytes=kv_bytes, canonical_tensors=len(atlas['canonical_tensors']),
                value_lifetimes=len(records), concurrent_value_pairs_checked=overlap_pairs,
                bank_profiles=atlas['bank_profiles'], duration_seconds=time.monotonic() - started,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Metadata ownership, dispatch addressing, coverage and lifetimes only. No routes, compiled SRAM, numerical execution or token performance admitted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.atlas, args.configs)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))
