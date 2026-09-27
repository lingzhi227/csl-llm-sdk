"""Fuse adjacent same-input projections without dropping rows or changing weights.

This lowers pure duplicate input preparation and exposes projection concurrency.
It does not admit operand multicast, return routing or a full executable schedule.
"""
import hashlib
import json
from pathlib import Path


def split_stream(start, count, ring):
    out = []; offset = 0
    while count:
        local = start % ring; take = min(count, ring - local)
        out.append(dict(class_start=local, count=take, slot=start // ring, bundle_tile_start=offset))
        start += take; offset += take; count -= take
    return out


def bundle_graph(graph, atlas):
    matrices = {m['tensor']: m for m in atlas['matrices']}
    bundles = []; events = []; producer = {}; consumed = set(); nodes = graph['nodes']; i = 0
    while i < len(nodes):
        first = nodes[i]; members = [first]
        if first['op'] in ('fp8_projection', 'bf16_projection'):
            matrix = matrices[first['weights'][0]]; end = matrix['stream_start'] + matrix['tiles']; j = i + 1
            while j < len(nodes):
                other = nodes[j]
                if other['op'] != first['op'] or other['inputs'] != first['inputs'] or other['attributes'] != first['attributes']:
                    break
                candidate = matrices[other['weights'][0]]
                if candidate['kind'] != matrix['kind'] or candidate['k_blocks'] != matrix['k_blocks'] or candidate['stream_start'] != end:
                    break
                members.append(other); end += candidate['tiles']; j += 1
        inputs = list(dict.fromkeys(v for n in members for v in n['inputs']))
        outputs = [v for n in members for v in n['outputs']]
        dependencies = sorted({producer[v] for v in inputs if v != 'token'})
        event = dict(id=len(events), original_nodes=[n['id'] for n in members], inputs=inputs, outputs=outputs,
                     dependencies=dependencies, data_dependencies=list(dependencies), memory_dependencies=[], op=first['op'])
        if first['op'] in ('fp8_projection', 'bf16_projection', 'embedding_lookup'):
            mm = [matrices[n['weights'][0]] for n in members]
            kind = mm[0]['kind']; ring = atlas['classes'][kind]['pes']; k = mm[0]['k_blocks']
            count = sum(m['tiles'] for m in mm); start = mm[0]['stream_start']
            if start % k or ring % k or atlas['classes'][kind]['phase'] % k:
                raise ValueError('Unaligned complete K groups')
            if len(mm) > 1 and count > ring:
                raise ValueError('Projections cannot run concurrently on the same bank PE')
            record = dict(id=len(bundles), event=event['id'], kind=kind, mode=mm[0]['mode'], input=inputs[0],
                          k_blocks=k, stream_start=start, tiles=count, output_rows=sum(m['shape'][0] for m in mm),
                          input_elements=mm[0]['shape'][1], profile=kind + '-k' + str(k),
                          matrices=[dict(matrix=m['id'], node=n['id'], tensor=m['tensor'], output=n['outputs'][0],
                                         bundle_row_start=(m['stream_start'] - start) // k * 2, rows=m['shape'][0])
                                    for m, n in zip(mm, members)],
                          segments=split_stream(start, count, ring),
                          same_input_preparation_shared=len(mm) > 1, participants_overlap_within_concurrent_segment=False,
                          segment_order='Different slots at the same physical owners are sequential; embedding dispatch selects only the requested row.')
            for m in mm:
                if m['id'] in consumed:
                    raise ValueError('Duplicate matrix binding')
                consumed.add(m['id'])
            bundles.append(record); event['bundle'] = record['id']; event['op'] = 'projection_bundle' if len(mm) > 1 else first['op']
        events.append(event)
        for output in outputs:
            if output in producer:
                raise ValueError('Duplicate producer')
            producer[output] = event['id']
        i += len(members)
    if consumed != {m['id'] for m in atlas['matrices']}:
        raise ValueError('Missing matrix')
    # Simultaneous bundle outputs must remain disjoint in the original closed
    # lifetime arena. All have a common lifetime before any bundled consumer.
    arena = atlas['value_arena']['values']
    pairs = 0
    for b in bundles:
        event = events[b['event']]; values = [arena[v] for v in event['inputs'] + event['outputs'] if v != 'token']
        for a, left in enumerate(values):
            for right in values[a + 1:]:
                if not (left['base_cell'] + left['reserved_cells'] <= right['base_cell'] or
                        right['base_cell'] + right['reserved_cells'] <= left['base_cell']):
                    raise ValueError('Simultaneous projection input/output alias')
                pairs += 1
    # Data dependencies alone cannot protect storage recycled by the serial
    # arena allocator. A later producer must wait for the last read of every
    # earlier value whose cells it overwrites. These are completion edges, not
    # send-callback edges, and remain necessary even between unrelated branches.
    node_event = {n: e['id'] for e in events for n in e['original_nodes']}
    aliases = 0
    records = list(arena.values())
    for i, a in enumerate(records):
        for b in records[i + 1:]:
            if max(a['base_cell'], b['base_cell']) >= min(a['base_cell'] + a['reserved_cells'], b['base_cell'] + b['reserved_cells']):
                continue
            before, after = (a, b) if a['birth'] < b['birth'] else (b, a)
            if before['last_use'] >= after['birth']:
                raise ValueError('Original live storage alias')
            reader = node_event[before['last_use']]; writer = node_event[after['birth']]
            if reader >= writer:
                raise ValueError('Fusion crosses a memory lifetime boundary')
            events[writer]['memory_dependencies'].append(reader); aliases += 1
    ancestors = []; raw_memory_edges = 0
    for e in events:
        candidates = sorted(set(e['memory_dependencies']), reverse=True)
        raw_memory_edges += len(candidates)
        covered = 0
        for d in e['data_dependencies']:
            covered |= ancestors[d] | (1 << d)
        kept = []
        for d in candidates:
            if not covered & (1 << d):
                kept.append(d); covered |= ancestors[d] | (1 << d)
        e['memory_dependencies'] = sorted(kept)
        e['dependencies'] = sorted(set(e['data_dependencies'] + kept))
        ancestors.append(covered)
    return dict(schema='wse-shared-input-projection-bundles-v1', model=graph['model'], revision=graph['revision'],
                bundles=bundles, events=events, value_producers=producer,
                metrics=dict(original_nodes=len(nodes), events=len(events), original_matrices=len(consumed), bundles=len(bundles),
                             original_fp8_projections=sum(n['op'] == 'fp8_projection' for n in nodes),
                             fp8_input_preparations=sum(b['kind'] == 'fp8' for b in bundles),
                             fp8_bundles=sum(b['kind'] == 'fp8' for b in bundles),
                             bf16_gemv_bundles=sum(b['kind'] == 'bf16' and b['mode'] == 'gemv' for b in bundles),
                             simultaneous_value_pairs_checked=pairs,
                             reused_storage_pairs=aliases,
                             memory_completion_edges_before_reduction=raw_memory_edges,
                             memory_completion_edges=sum(len(e['memory_dependencies']) for e in events),
                             maximum_concurrent_tiles=max((b['tiles'] for b in bundles if len(b['matrices']) > 1), default=0)),
                physical_routes_admitted=False, full_model_executable=False, full_model_speed_target_achieved=False,
                scope='Full original tensor/output identity and pure shared-input preparation lowering. Reduction topology is a separate numerical implementation choice; input multicast, return routing and actual composed resources remain unqualified.')


def build(root, atlas_path):
    root = Path(root); atlas_path = Path(atlas_path); graph_path = root / 'configs/model-graph.json'
    graph = json.loads(graph_path.read_text()); atlas = json.loads(atlas_path.read_text())
    if hashlib.sha256(graph_path.read_bytes()).hexdigest() != atlas['provenance']['model-graph.json']:
        raise ValueError('Model graph changed')
    result = bundle_graph(graph, atlas)
    result['provenance'] = dict(atlas_sha256=hashlib.sha256(atlas_path.read_bytes()).hexdigest(),
                                graph_sha256=atlas['provenance']['model-graph.json'])
    return result
