"""Root broadcasts -> bounded packet merges ->16 shared-Q/K consumer groups.

Horizontal colors0/1/19 have a single producer per installed route. Vertical
colors3/9 alternate at software merge points: there is never a router with
simultaneous RAMP/cardinal RX. The rails bypass norm cohosts without using their
queues. All traffic remains inside the original mixer rectangle.
"""
from collections import Counter, defaultdict
from spatial.layer_mlp_network import broadcast
from spatial.layer_routes import direction

HORIZONTAL = (0, 1, 19)
VERTICAL = (3, 9)
DELTA = {'NORTH': (0, -1), 'SOUTH': (0, 1), 'EAST': (1, 0), 'WEST': (-1, 0)}
OPPOSITE = {'NORTH': 'SOUTH', 'SOUTH': 'NORTH', 'EAST': 'WEST', 'WEST': 'EAST'}


def frontend_index(matrix, row):
    if type(matrix) is not int or type(row) is not int or not 0 <= matrix < 4:
        raise ValueError('Original frontend projection coordinate')
    size = (10240, 6144, 48, 48)[matrix]
    if not 0 <= row < size or (matrix < 2 and row % 2):
        raise ValueError('Original BF16 output row/pair required')
    if matrix == 0:
        group = row//128 if row < 2048 else (row-2048)//128 if row < 4096 else (row-4096)//384
        return group, row
    if matrix == 1:
        return row//384, 10240+row
    return row//3, 16384+(matrix-2)*48+row


def lower_frontend_network(region, profiles, setups, *, columns=tuple(range(74, 90)), prune_unused=False, native_frames=False, batch_rows=False):
    if native_frames and not prune_unused: raise ValueError('Native frames require explicit consumer ownership')
    if batch_rows and not native_frames: raise ValueError('Batched planes require native packet ownership')
    if region['rect'] != [63, 67, 45, 79] or len(columns) != 16 or len(set(columns)) != 16:
        raise ValueError('Selected complete layer00 mixer geometry required')
    if any(type(x) is not int or not 63 <= x < 108 for x in columns):
        raise ValueError('Consumer column outside mixer')
    by_pe = {tuple(p['pe']): p for p in profiles}
    if len(by_pe) != len(profiles): raise ValueError('Duplicate cohost')
    consumers = [dict(group=g, pe=[x, 145], incoming=3) for g, x in enumerate(columns)]
    if any(by_pe[tuple(c['pe'])]['source'] != 'device_mixer_layer_mlp_standby.csl' for c in consumers):
        raise ValueError('Consumer requires the original internal standby cohost')
    roots = {tuple(pe): descriptors for pe, descriptors in setups if descriptors[1] and descriptors[4] == 0}
    if len(roots) != 88: raise ValueError('All88 original QKV roots required')
    for pe, descriptors in setups:
        if any(descriptors[m*8+1] and descriptors[m*8+4] == 0 for m in range(4)) and tuple(pe) not in roots:
            raise ValueError('Another projection has an unbound source')
    targets = {}
    for pe, descriptors in roots.items():
        groups = set()
        for matrix in range(4):
            _, count, first, stride, key, _, _, _ = descriptors[matrix*8:matrix*8+8]
            if key == 0:
                groups.update(frontend_index(matrix, first+i*stride)[0] for i in range(count))
        targets[pe] = sorted(groups) if prune_unused else list(range(16))
    by_row = defaultdict(list)
    if batch_rows:
        ordinary=sorted((pe for pe in roots if pe[1]<142),key=lambda pe:(pe[1],pe[0]))
        for start in range(0,len(ordinary),3):
            batch=ordinary[start:start+3];by_row[max(pe[1] for pe in batch)].extend(batch)
        by_row[142].extend(pe for pe in roots if pe[1]>=142)
    else:
        for pe in roots: by_row[min(pe[1], 142)].append(pe)
    installed = {}
    producers = []
    horizontal_at = defaultdict(set)

    def add(record):
        key = (*record['pe'], record['color'])
        if key in installed: raise ValueError('Two frontend routes claim the same PE/color')
        installed[key] = record

    for y, sources in sorted(by_row.items()):
        sources.sort(key=lambda p: (p[1], p[0]))
        if len(sources) > 3: raise ValueError('Horizontal source planes exhausted')
        for color, source in zip(HORIZONTAL, sources):
            origin = (source[0], y)
            row_routes = broadcast(origin, {(columns[g], y): g for g in targets[source]}, color)
            if native_frames:
                for r in row_routes: r.pop('filter_tag', None)
            if source != origin:
                below=source[1]>y;step=-1 if below else 1
                for r in row_routes:
                    if tuple(r['pe']) == origin: r['rx'] = ['SOUTH' if below else 'NORTH']
                for yy in range(source[1], y, step):
                    add(dict(pe=[source[0], yy], color=color, rx=['RAMP' if yy == source[1] else 'SOUTH' if below else 'NORTH'], tx=['NORTH' if below else 'SOUTH']))
            for r in row_routes: add(r)
            for g in targets[source]: horizontal_at[columns[g], y].add(color)
            producer = dict(pe=list(source), color=color, row=y)
            if prune_unused: producer['groups'] = targets[source]
            producers.append(producer)
    mergers = []
    if prune_unused:
        for group, x in enumerate(columns):
            rows = [y for y in range(67, 143) if horizontal_at[x, y]]
            if not rows: raise ValueError('Missing complete frontend group')
            previous = None
            for index, y in enumerate(rows):
                incoming = VERTICAL[(len(rows)-index) % 2]
                outgoing = 12-incoming
                if by_pe[x, y]['source'] != 'device_mixer_layer_mlp_standby.csl':
                    raise ValueError('A packet merge would consume a norm cohost queue')
                if previous is not None:
                    for yy in range(previous+1, y):
                        add(dict(pe=[x, yy], color=incoming, rx=['NORTH'], tx=['SOUTH']))
                    add(dict(pe=[x, y], color=incoming, rx=['NORTH'], tx=['RAMP']))
                add(dict(pe=[x, y], color=outgoing, rx=['RAMP'], tx=['SOUTH']))
                mergers.append(dict(pe=[x, y], group=group, horizontal=sorted(horizontal_at[x, y]),
                                    vertical=previous is not None, incoming=incoming, outgoing=outgoing))
                previous = y
            for y in range(rows[-1]+1, 145):
                add(dict(pe=[x, y], color=3, rx=['NORTH'], tx=['SOUTH']))
            add(dict(pe=[x, 145], color=3, rx=['NORTH'], tx=['RAMP']))
    else:
        for y in range(67, 143):
          for group, x in enumerate(columns):
            if not horizontal_at[x, y]: raise ValueError('Missing source row')
            if by_pe[x, y]['source'] != 'device_mixer_layer_mlp_standby.csl':
                raise ValueError('A packet merge would consume a norm cohost queue')
            incoming = VERTICAL[(y-67) % 2]
            outgoing = VERTICAL[1-(y-67) % 2]
            if y > 67: add(dict(pe=[x, y], color=incoming, rx=['NORTH'], tx=['RAMP']))
            add(dict(pe=[x, y], color=outgoing, rx=['RAMP'], tx=['SOUTH']))
            mergers.append(dict(pe=[x, y], group=group, horizontal=sorted(horizontal_at[x, y]),
                                vertical=y > 67, incoming=incoming, outgoing=outgoing))
        for consumer in consumers:
            x, _ = consumer['pe']
            for y in (143, 144): add(dict(pe=[x, y], color=3, rx=['NORTH'], tx=['SOUTH']))
            add(dict(pe=consumer['pe'], color=3, rx=['NORTH'], tx=['RAMP']))
    plan = dict(schema='original-mixer-frontend-network-v1', rect=region['rect'],
                producers=producers, mergers=mergers, consumers=consumers, routes=list(installed.values()),
                wire_words=5, protocol='invocation32,index16,bf16,bf16 in five payload16 wavelets; horizontal tag identifies Q/K group',
                source_release='Outgoing frame copied by DMA; downstream completion is separate',
                complete_input_packets_per_group=518, recurrent_core_packets_per_group=192,
                compiled=False, executed=False, physical=False, complete_neural_stage=False)
    if prune_unused: plan['schema'] = 'original-mixer-frontend-network-v2'
    if native_frames:
        plan['schema'] = 'original-mixer-frontend-network-v3'
        plan['protocol'] = 'Five full32-bit words: token,row,rows,packed-BF16,matrix; destination checks original semantic ownership'
        copies = Counter()
        for pe, descriptors in roots.items():
            packets = sum(descriptors[m*8+1] for m in range(4))
            for group in targets[pe]: copies[group] += packets
        plan['consumer_projected_packets'] = [copies[g] for g in range(16)]
        plan['batched_source_rows'] = batch_rows
    plan['audit'] = audit_frontend_network(plan, setups)
    return plan


def audit_frontend_network(plan, setups):
    if plan['rect'] != [63, 67, 45, 79] or plan['wire_words'] != 5:
        raise ValueError('Frontend network geometry/protocol changed')
    native = plan['schema'] == 'original-mixer-frontend-network-v3'
    sparse = native or plan['schema'] == 'original-mixer-frontend-network-v2'
    if plan['schema'] not in ('original-mixer-frontend-network-v1', 'original-mixer-frontend-network-v2', 'original-mixer-frontend-network-v3'):
        raise ValueError('Unknown frontend protocol')
    installed = {}
    for r in plan['routes']:
        x, y = r['pe']; key = (x, y, r['color'])
        if (not 63 <= x < 108 or not 67 <= y < 146 or key in installed
                or r['color'] not in HORIZONTAL+VERTICAL or len(r['rx']) != 1
                or len(set(r['tx'])) != len(r['tx'])):
            raise ValueError('Frontend route ownership or single-RX rule violated')
        installed[key] = r
        if native and ('filter_tag' in r or 'filter_max' in r): raise ValueError('Native packet payload cannot be tag filtered')
    mergers = {tuple(m['pe']): m for m in plan['mergers']}
    consumers = {tuple(c['pe']): c for c in plan['consumers']}
    if ((not sparse and len(mergers) != 1216) or not 0 < len(mergers) <= 1216
            or len(mergers) != len(plan['mergers']) or len(consumers) != 16 or len(plan['consumers']) != 16):
        raise ValueError('Incomplete merged consumer geometry')
    if sorted(c['group'] for c in consumers.values()) != list(range(16)):
        raise ValueError('Missing/aliased Q/K consumer group')
    producers = {tuple(p['pe']): p for p in plan['producers']}
    expected = {tuple(pe) for pe, s in setups if s[1] and s[4] == 0}
    if len(producers) != len(plan['producers']) or set(producers) != expected:
        raise ValueError('Original producer coverage changed')
    descriptors_by_pe = {tuple(pe): s for pe, s in setups}
    maximum_hops = 0; paths = 0; wire_copies = Counter()
    for source, producer in producers.items():
        needed = set(range(16))
        if sparse:
            needed = set(); descriptors = descriptors_by_pe[source]
            for matrix in range(4):
                _, count, first, stride, key, _, _, _ = descriptors[matrix*8:matrix*8+8]
                if key == 0:
                    needed.update(frontend_index(matrix, first+i*stride)[0] for i in range(count))
            if producer.get('groups') != sorted(needed): raise ValueError('Consumer pruning changes original output ownership')
        paths += len(needed)
        if native:
            for group in needed: wire_copies[group] += sum(descriptors[m*8+1] for m in range(4))
        for group in ([None] if native else range(16)):
            pending = [(source, producer['color'], 'RAMP', 0)]
            seen = set(); delivered = []
            while pending:
                pe, color, incoming, hops = pending.pop()
                key = (*pe, color)
                route = installed.get(key)
                if key in seen or route is None or route['rx'] != [incoming]:
                    raise ValueError('Packet route loops, drops, or receives from the wrong port')
                seen.add(key)
                for tx in route['tx']:
                    if tx == 'RAMP':
                        if 'filter_tag' in route and route['filter_tag'] != group: continue
                        if pe in consumers:
                            if color != consumers[pe]['incoming'] or (not native and consumers[pe]['group'] != group):
                                raise ValueError('Packet reached the wrong consumer')
                            delivered.append(pe); maximum_hops = max(maximum_hops, hops)
                        else:
                            merger = mergers.get(pe)
                            if merger is None or (not native and merger['group'] != group) or color not in merger['horizontal']+([merger['incoming']] if merger['vertical'] else []):
                                raise ValueError('Software queue/color binding mismatch')
                            pending.append((pe, merger['outgoing'], 'RAMP', hops))
                    else:
                        dx, dy = DELTA[tx]
                        pending.append(((pe[0]+dx, pe[1]+dy), color, OPPOSITE[tx], hops+1))
            if native:
                if len(delivered)!=len(needed) or {consumers[p]['group'] for p in delivered}!=needed:
                    raise ValueError('Native multicast changes exact consumer ownership')
            elif len(delivered) != int(group in needed): raise ValueError('Source packet is missing or duplicated at a consumer')
    if native and plan.get('consumer_projected_packets')!=[wire_copies[g] for g in range(16)]:
        raise ValueError('Native transport drain count differs from full source packet extents')
    values = [Counter() for _ in range(4)]
    groups = Counter()
    for pe, descriptors in setups:
        if tuple(pe) not in producers: continue
        for matrix in range(4):
            _, count, first, stride, key, parts, bf16, rows = descriptors[matrix*8:matrix*8+8]
            if key != 0 or parts != 40 or rows != (1 if matrix >= 2 else 2) or bf16 != int(matrix >= 2):
                raise ValueError('Original root descriptor contract changed')
            for iteration in range(count):
                row = first+iteration*stride
                group, _ = frontend_index(matrix, row)
                groups[group] += 1
                for i in range(rows): values[matrix][row+i] += 1
    for count, size in zip(values, (10240, 6144, 48, 48)):
        if count != Counter({i: 1 for i in range(size)}): raise ValueError('Original frontend output omitted or duplicated')
    if groups != Counter({i: 518 for i in range(16)}): raise ValueError('Wrong per-group packet cardinality')
    return dict(passed=True, original_projection_values=16480, projected_packets=8288,
                projected_wire_words=41440, root_to_consumer_paths=paths,
                maximum_fabric_hops=maximum_hops, merge_pes=len(mergers), consumer_groups=16,
                no_host_gather=True, physical=False, numerical_qualified=False)
