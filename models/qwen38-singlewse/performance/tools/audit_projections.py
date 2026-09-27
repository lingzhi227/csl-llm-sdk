"""Independent graph/arena/route oracle for the complete projection lowering.

Imports the candidate only to obtain its dense route output, never as the oracle.
Dense route words stay remote; they are abstract ports, not SDK register images.
"""
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def edges(k):
    """Enumerate balanced preorder intervals with a separate iterative oracle."""
    pending = [(0, k, 0)]; result = []
    while pending:
        root, size, depth = pending.pop()
        remaining = size - 1
        left = (remaining + 1) // 2; right = remaining // 2
        for side, offset, length in [(0, 1, left), (1, left + 1, right)]:
            if length:
                child = root + offset
                result.append((root, child, 2 * depth + side))
                pending.append((child, length, depth + 1))
    require(len(result) == k - 1, 'Tree edge conservation')
    require(sorted(child for _, child, _ in result) == list(range(1, k)), 'Every non-root has one parent')
    return result


def graph_audit(graph, atlas, plan):
    events = plan['events']; nodes = graph['nodes']; matrices = atlas['matrices']
    require(plan['model'] == graph['model'] and plan['revision'] == graph['revision'], 'Model identity')
    require([e['id'] for e in events] == list(range(len(events))), 'Event numbering')
    require([n for e in events for n in e['original_nodes']] == [n['id'] for n in nodes], 'All original nodes exactly once and ordered')
    indexed = {n['id']: n for n in nodes}; owner = {}; producer = {}; last_use = {}; closure = []
    for e in events:
        i = e['id']; members = [indexed[n] for n in e['original_nodes']]
        require(e['inputs'] == list(dict.fromkeys(v for n in members for v in n['inputs'])), 'All original inputs')
        require(e['outputs'] == [v for n in members for v in n['outputs']], 'All original outputs')
        expected = sorted({producer[v] for v in e['inputs'] if v != 'token'})
        require(e['data_dependencies'] == expected, 'Exact original data dependencies')
        require(e['dependencies'] == sorted(set(expected + e['memory_dependencies'])), 'Dependency union')
        ancestors = set()
        for d in e['dependencies']:
            require(0 <= d < i, 'Dependency is not an earlier event')
            ancestors.add(d); ancestors.update(closure[d])
        closure.append(ancestors)
        for n in e['original_nodes']:
            owner[n] = i
        for v in e['outputs']:
            require(v not in producer, 'Multiple producers')
            producer[v] = last_use[v] = i
        for v in e['inputs']:
            if v != 'token':
                last_use[v] = i
        if len(members) > 1:
            require(all(n['op'] == members[0]['op'] and n['inputs'] == members[0]['inputs']
                        and n['attributes'] == members[0]['attributes'] for n in members), 'Illegal fusion')
            require(members[0]['op'] in ('fp8_projection', 'bf16_projection'), 'Only pure projections may fuse')
    require(producer == plan['value_producers'], 'Producer map')
    last_use[graph['selected_token']] = len(events)
    values = list(atlas['value_arena']['values'].items()); aliases = 0
    # Independent set-based reachability, with lifetimes rebuilt from event I/O.
    for i, (an, a) in enumerate(values):
        for bn, b in values[i + 1:]:
            if max(a['base_cell'], b['base_cell']) >= min(a['base_cell'] + a['reserved_cells'], b['base_cell'] + b['reserved_cells']):
                continue
            earlier, later = (an, bn) if producer[an] < producer[bn] else (bn, an)
            require(last_use[earlier] < producer[later], 'Fusion extends a value over live reused storage')
            require(last_use[earlier] in closure[producer[later]], 'Missing storage completion ordering')
            aliases += 1
    consumed = []; total_tiles = 0; segments = 0
    for b in plan['bundles']:
        e = events[b['event']]; mm = [matrices[m['matrix']] for m in b['matrices']]
        require(e['bundle'] == b['id'] and b['input'] == e['inputs'][0], 'Bundle event identity')
        require(b['profile'] == b['kind'] + '-k' + str(b['k_blocks']), 'Profile identity')
        require(b['input_elements'] == mm[0]['shape'][1], 'Full input extent')
        prefix = 0; rows = 0
        for binding, m in zip(b['matrices'], mm):
            n = indexed[binding['node']]
            require(n['weights'][0] == m['tensor'] == binding['tensor'] and n['outputs'][0] == binding['output'], 'Original tensor/output identity')
            require(m['stream_start'] == b['stream_start'] + prefix and m['kind'] == b['kind']
                    and m['k_blocks'] == b['k_blocks'] and m['mode'] == b['mode'], 'Exact contiguous matrix extent')
            require(binding['rows'] == m['shape'][0] and binding['bundle_row_start'] == rows, 'All output rows preserved')
            prefix += m['tiles']; rows += m['shape'][0]; consumed.append(m['id'])
        require((prefix, rows) == (b['tiles'], b['output_rows']), 'Bundle conservation')
        c = atlas['classes'][b['kind']]; k = b['k_blocks']; offset = 0
        require(c['pes'] % k == c['phase'] % k == 0, 'Profile groups change under class phase')
        require(len(mm) == 1 or prefix <= c['pes'], 'Concurrent matrix bank alias')
        for s in b['segments']:
            start = b['stream_start'] + offset
            require(s['bundle_tile_start'] == offset and s['class_start'] == start % c['pes']
                    and s['slot'] == start // c['pes'] and s['count'] == min(prefix - offset, c['pes'] - start % c['pes']), 'Exact resident slot dispatch')
            require(s['class_start'] % k == s['count'] % k == 0, 'Partial reduction group')
            offset += s['count']; segments += 1
        require(offset == prefix, 'Complete bundle dispatch')
        total_tiles += prefix
    require(sorted(consumed) == list(range(len(matrices))), 'Every original matrix exactly once')
    return dict(nodes=len(nodes), events=len(events), original_matrices=len(consumed), bundles=len(plan['bundles']),
                original_matrix_tiles=total_tiles, complete_segments=segments, reused_storage_pairs=aliases,
                all_reused_storage_ordered=True, memory_completion_edges=sum(len(e['memory_dependencies']) for e in events))


def audit(root):
    from spatial.forest import lower, path_embedding
    started = time.monotonic(); root = Path(root)
    atlas_path = root / 'model-atlas.json'; graph_path = root / 'configs/model-graph.json'
    atlas = json.loads(atlas_path.read_text()); graph = json.loads(graph_path.read_text())
    plan = json.loads((root / 'bundles.json').read_text()); doc = json.loads((root / 'forests.json').read_text())
    require(plan['provenance']['atlas_sha256'] == doc['atlas_sha256'] == digest(atlas_path), 'Atlas provenance')
    require(plan['provenance']['graph_sha256'] == atlas['provenance']['model-graph.json'] == digest(graph_path), 'Graph provenance')
    result = graph_audit(graph, atlas, plan)
    embedding = path_embedding(atlas); xy, bank_positions, _, _ = embedding
    width, height, count = (atlas['policy'][k] for k in ('width', 'height', 'bank_pes'))
    require(np.all((xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)), 'Physical bounds')
    require(len(np.unique(xy[:, 1] * width + xy[:, 0])) == len(xy), 'Physical path coordinate reuse')
    require(np.all(np.abs(np.diff(xy, axis=0)).sum(axis=1) == 1), 'Non-neighbor hop')
    rows = np.array([y for y in range(height) if y not in atlas['geometry']['actor_rows']])
    row, col = np.divmod(np.arange(count), width)
    require(np.array_equal(xy[bank_positions, 0], np.where(row % 2, width - 1 - col, col))
            and np.array_equal(xy[bank_positions, 1], rows[row]), 'Independent full bank geometry')
    is_bank = np.zeros(len(xy), bool); is_bank[bank_positions] = True
    require(np.all(np.isin(xy[~is_bank, 1], atlas['geometry']['actor_rows'])) and np.all(np.isin(xy[~is_bank, 0], [0, width - 1])), 'Actor bridge geometry')
    ports = {'NONE': 0, 'RAMP': 1, 'NORTH': 2, 'SOUTH': 3, 'WEST': 4, 'EAST': 5}
    require(doc['ports'] == ports and doc['colors'] == list(range(3, 16)), 'Abstract port/color schema')
    delta = np.diff(xy, axis=0); back = np.zeros(len(xy), np.uint16); forward = np.zeros_like(back)
    for x, y, f, r in [(1, 0, 5, 4), (-1, 0, 4, 5), (0, 1, 3, 2), (0, -1, 2, 3)]:
        selected = (delta[:, 0] == x) & (delta[:, 1] == y)
        forward[:-1][selected] = f; back[1:][selected] = r
    excluded = np.zeros((height, width), bool)
    for rectangle in [g['state_rectangle'] for g in atlas['gdn']] + [atlas['state_class_spare_rectangle']]:
        x, y, w, h = rectangle; excluded[y:y+h, x:x+w] = True
    eligible_positions = bank_positions[~excluded[xy[bank_positions, 1], xy[bank_positions, 0]]]
    profiles = []; files = {}; expected_profiles = [('fp8', 40), ('fp8', 48), ('fp8', 136), ('bf16', 40)]
    require([(p['kind'], p['k_blocks']) for p in doc['profiles']] == expected_profiles, 'Four complete profiles')
    for profile, (kind, k) in zip(doc['profiles'], expected_profiles):
        words, nodes = lower(atlas, kind, k, embedding)
        expected_nodes = bank_positions if kind == 'fp8' else eligible_positions
        require(np.array_equal(nodes, expected_nodes), 'Class eligibility')
        require(words.shape == (len(xy), 13), 'Dense route shape')
        grouped = nodes.reshape(-1, k); relative = edges(k); active_total = 0; max_hops = 0; forwarding_holes = 0
        compact = [dict(active=0, source=0, destination=0) for _ in range(k)]; cuts = [0] * (k + 1)
        for parent, child, plane in relative:
            bit = 1 << plane
            compact[parent]['destination'] |= bit; compact[child]['source'] |= bit
            for rank in range(parent, child + 1):
                require(not compact[rank]['active'] & bit, 'Compact PE/color collision'); compact[rank]['active'] |= bit
            for rank in range(parent + 1, child + 1):
                cuts[rank] |= bit
        require(profile['vertices'] == compact and profile['cuts'] == cuts and profile['colors'] == list(range(3, 16)), 'Compact published profile differs from oracle')
        for color in range(13):
            differences = np.zeros(len(xy) + 1, np.int16)
            sources = np.zeros(len(xy), bool); destinations = np.zeros(len(xy), bool)
            for parent, child, plane in relative:
                if plane != color:
                    continue
                first, last = grouped[:, parent], grouped[:, child]
                np.add.at(differences, first, 1); np.add.at(differences, last + 1, -1)
                sources[last] = True; destinations[first] = True
                max_hops = max(max_hops, int(np.max(last - first)))
            occupancy = np.cumsum(differences[:-1]); require(np.all((occupancy == 0) | (occupancy == 1)), 'Independent PE/color collision')
            active = occupancy == 1
            expected = np.where(active, np.where(sources, 1, forward) | (np.where(destinations, 1, back) << 3), 0)
            require(np.array_equal(words[:, color], expected), 'Dense route differs from independent edge expansion')
            active_total += int(active.sum()); forwarding_holes += int(np.count_nonzero(active & ~np.isin(np.arange(len(xy)), nodes)))
        name = kind + '-k' + str(k)
        path = root / (name + '.npy')
        with path.open('xb') as f:
            np.save(f, words, allow_pickle=False)
        files[path.name] = dict(bytes=path.stat().st_size, sha256=digest(path))
        profiles.append(dict(name=name, contributing_pes=len(nodes), independent_trees=len(grouped),
                             tree_edges=len(grouped) * (k - 1), route_entries_checked=int(words.size),
                             active_pe_color_entries=active_total, forwarding_hole_entries=forwarding_holes,
                             maximum_single_edge_physical_hops=max_hops, abstract_routes_match=True))
    return dict(schema='wse-projection-forest-audit-v1', passed=True, physical=False, weights_read=False,
                graph=result, profiles=profiles, physical_path_pes=len(xy), actor_bridge_pes=int((~is_bank).sum()),
                dense_abstract_route_files=files, atlas_sha256=digest(atlas_path),
                bundles_sha256=digest(root / 'bundles.json'), forests_sha256=digest(root / 'forests.json'),
                duration_seconds=time.monotonic() - started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                input_multicast_admitted=False, output_scatter_admitted=False, route_epochs_admitted=False,
                compiled_sram_admitted=False, full_model_executable=False, full_model_speed_target_achieved=False,
                scope='Complete original projection identities and arena-safe event DAG; four collision-free abstract reduction forests. No operand/return transport, runtime route changes, numerical execution or model throughput qualified.')
