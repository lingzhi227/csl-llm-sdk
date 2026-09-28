"""Static paired-color routes between original MLP reductions and fusion actors.

P23 matrix/auxiliary addresses and P24 arithmetic trees are unchanged. This
overlay recolors interval edges and assigns consecutive original quantization
groups to nearby low-payload actors. A separate early boundary buffer is required
for the following producer: one active fusion group alone would serialize roots.
No multi-input router merge, route reconfiguration or PE software relay is used.
"""
from collections import Counter
from copy import deepcopy
import heapq
from spatial.layer_routes import contraction_groups, reduction_routes, direction
from spatial.layer_schedule import rank_xy

COLORS = (0, 1, *range(3, 18), 19)
DELTA = {'EAST': (1, 0), 'WEST': (-1, 0), 'SOUTH': (0, 1), 'NORTH': (0, -1)}
OPPOSITE = {'EAST': 'WEST', 'WEST': 'EAST', 'SOUTH': 'NORTH', 'NORTH': 'SOUTH'}


def compact_tree_groups(region):
    """Optimal coloring of closed rank intervals; touching endpoints conflict."""
    for group in contraction_groups(region):
        ends = {}
        intervals = sorted((min(n['rank'], n['parent']), max(n['rank'], n['parent']), n['rank'])
                           for n in group['nodes'] if n['parent'] is not None)
        for first, last, rank in intervals:
            color = next(c for c in range(3, 18) if ends.get(c, -1) < first)
            ends[color] = last
            group['nodes'][rank]['route_color'] = color
        yield group


def assign_actors(region, groups):
    """Monotone minimum-Manhattan matching to distinct bottom-row actors.

    An actor owns groups whose first row belongs to its producer. Only the
    immediately following producer can contribute the final group's suffix.
    Weight owners, pages and native K slices do not change.
    """
    roots = {tuple(g['root_pe']) for g in groups}
    bottom = region['rect'][1] + region['rect'][3] - 1
    candidates = sorted((a for a in region['quantization_actors']
                         if tuple(a['pe']) not in roots and a['pe'][1] == bottom), key=lambda a: tuple(a['pe']))
    order = sorted(groups, key=lambda g: tuple(g['root_pe']))
    if len(candidates) < len(order):
        raise ValueError('Insufficient distinct low-payload non-root actors')
    dp = {(0, j): (0, []) for j in range(len(candidates) + 1)}
    for i in range(1, len(order) + 1):
        for j in range(i, len(candidates) + 1):
            cost = sum(abs(a - b) for a, b in zip(order[i-1]['root_pe'], candidates[j-1]['pe']))
            take = (dp[i-1, j-1][0] + cost, dp[i-1, j-1][1] + [j-1])
            dp[i, j] = min(take, dp.get((i, j-1), (10**9, [])))
    selected = {g['index']: candidates[a] for g, a in zip(order, dp[len(order), len(candidates)][1])}
    owners = {}
    for q in range(region['matrices'][0]['shape'][0] // 128):
        matches = [g['index'] for g in groups
                   if g['output_start'] * g['rows'] <= 128*q < (g['output_start'] + g['output_count']) * g['rows']]
        if len(matches) != 1:
            raise ValueError('Original quantization group has no unique first-row producer')
        owners[q] = matches[0]
    actors = []
    for group in groups:
        actor = dict(selected[group['index']])
        actor.update(producer=group['index'], groups=[q for q, owner in owners.items() if owner == group['index']],
                     early_boundary_bytes=256, projection_receive_bytes=88)
        actors.append(actor)
    return actors, owners


def credit_fanout(region, group):
    """One source, static multicast along the two halves of its rank interval."""
    records = []
    for rank in range(group['workers']):
        pe = rank_xy(region, group['rank_start'] + rank)
        if rank == group['root']:
            incoming = 'RAMP'
            neighbors = [v for v in [rank-1, rank+1] if 0 <= v < group['workers']]
            outgoing = [direction(pe, rank_xy(region, group['rank_start'] + v)) for v in neighbors]
        else:
            step = 1 if rank < group['root'] else -1
            incoming = direction(pe, rank_xy(region, group['rank_start'] + rank + step))
            outgoing = ['RAMP']
            if 0 <= rank-step < group['workers']:
                outgoing.append(direction(pe, rank_xy(region, group['rank_start'] + rank-step)))
        records.append(dict(pe=pe, color=18, rx=incoming, tx=outgoing))
    return records


def shortest_route(source, destination, color, rect, used, pressure, endpoints):
    """Deterministic A* on (x,y,color), with only hardware xor-one swapping.

    Existing PE/color definitions are immutable. Endpoints are reserved against
    unrelated transit. Congestion is an explicit adjustable cost, not a timing
    prediction; its square avoids filling the same narrow bottom access lanes.
    """
    source, destination = tuple(source), tuple(destination)
    if source in used[color]:
        return None
    first = (*source, color)
    heuristic = lambda p: abs(p[0]-destination[0]) + abs(p[1]-destination[1])
    frontier = [(heuristic(source), 0, 0, first)]
    parent, distance = {first: None}, {first: (0, 0)}
    x, y, width, height = rect
    while frontier:
        _, length, swaps, at = heapq.heappop(frontier)
        if distance[at] != (length, swaps):
            continue
        if at[:2] == destination:
            path = []
            while at is not None:
                path.append(at); at = parent[at]
            return path[::-1]
        neighbors = [(at[0]+1, at[1]), (at[0]-1, at[1]), (at[0], at[1]+1), (at[0], at[1]-1)]
        neighbors.sort(key=heuristic)
        for xy in neighbors:
            if not (x <= xy[0] < x+width and y <= xy[1] < y+height):
                continue
            if xy != destination and xy in endpoints:
                continue
            for next_color in (at[2], at[2] ^ 1):
                nxt = (*xy, next_color)
                cost = (length + 1 + pressure[xy]**2, swaps + (next_color != at[2]))
                if next_color in used and xy not in used[next_color] and cost < distance.get(nxt, (10**9, 10**9)):
                    parent[nxt] = at; distance[nxt] = cost
                    heapq.heappush(frontier, (cost[0] + heuristic(xy), *cost, nxt))
    return None


def lower_path(path):
    records = []
    for i, at in enumerate(path):
        tx = 'RAMP' if i+1 == len(path) else direction(at[:2], path[i+1][:2])
        swap = i+1 < len(path) and at[2] != path[i+1][2]
        records.append(dict(pe=list(at[:2]), color=at[2],
                            rx='RAMP' if i == 0 else direction(at[:2], path[i-1][:2]), tx=[tx],
                            color_swap_x=bool(swap and tx in ('EAST', 'WEST')),
                            color_swap_y=bool(swap and tx in ('NORTH', 'SOUTH'))))
    return records


def build_fused_routes(region):
    if region['role'] != 'gate_up':
        raise ValueError('Original gate/up region required')
    groups = list(compact_tree_groups(region)); actors, owners = assign_actors(region, groups)
    used = {c: set() for c in COLORS}
    trees, fanouts = [], []
    for group in groups:
        for tree in reduction_routes(region, group):
            records = [dict(**r, color=tree['color']) for r in tree['routes']]
            for r in records:
                xy = tuple(r['pe'])
                if xy in used[r['color']]:
                    raise ValueError('Interval coloring collided')
                used[r['color']].add(xy)
            trees.append(dict(group=group['index'], source=tree['source'], destination=tree['destination'], routes=records))
        fanouts.extend(credit_fanout(region, group))
    flows = []
    for group in groups:
        first = group['output_start'] * group['rows']; end = first + group['output_count'] * group['rows']
        for owner in sorted({owners[row//128] for row in range(first, end, group['rows'])}):
            common = dict(producer=group['index'], actor=owner,
                          iterations=[i for i in range(group['output_count']) if owners[(first+i*group['rows'])//128] == owner])
            flows.extend([dict(**common, kind='projection', source=group['root_pe'], destination=actors[owner]['pe']),
                          dict(**common, kind='credit', source=actors[owner]['pe'], destination=group['root_pe'])])
    endpoints = {tuple(p) for f in flows for p in [f['source'], f['destination']]}
    pressure = Counter(pe for busy in used.values() for pe in busy)
    routed = []
    for flow in sorted(flows, key=lambda f: -sum(abs(a-b) for a, b in zip(f['source'], f['destination']))):
        choices = []
        for color in COLORS:
            path = shortest_route(flow['source'], flow['destination'], color, region['rect'], used, pressure, endpoints)
            if path:
                choices.append((sum(1+pressure[p[:2]]**2 for p in path),
                                sum(a[2] != b[2] for a, b in zip(path, path[1:])), color, path))
        if not choices:
            raise ValueError('No static route after %s flows: %r' % (len(routed), flow))
        *_, path = min(choices)
        for x, y, color in path:
            used[color].add((x, y)); pressure[x, y] += 1
        routed.append(dict(**flow, routes=lower_path(path)))
    result = dict(region=region['id'], rect=region['rect'], actors=actors, flows=routed,
                  reduction_trees=trees, credit_fanouts=fanouts,
                  matrix_and_auxiliary_addresses_changed=False, input_distribution_connected=False,
                  downstream_distribution_connected=False, executed=False, physical=False)
    result['audit'] = audit_fused_routes(region, result)
    return result


def audit_fused_routes(region, result):
    """Follow installed single-source routes, including swaps and multicast."""
    installed = {}; stats = Counter(); groups = list(compact_tree_groups(region))
    allowed = {a['rank']: a for a in region['quantization_actors']}
    assigned = Counter(); all_quantization = []
    for actor in result['actors']:
        original = allowed.get(actor['rank'])
        if original is None or actor['pe'] != original['pe'] or actor['resident_payload_bytes'] != original['resident_payload_bytes']:
            raise ValueError('Fusion actor moved to an unadmitted original bank')
        assigned[actor['rank']] += 1; all_quantization.extend(actor['groups'])
    if len(assigned) != len(groups) or set(assigned.values()) != {1} or sorted(all_quantization) != list(range(136)):
        raise ValueError('Fusion actor/group coverage')
    expected_edges = Counter((g['index'], tuple(rank_xy(region, g['rank_start']+n['rank'])),
                              tuple(rank_xy(region, g['rank_start']+n['parent'])))
                             for g in groups for n in g['nodes'] if n['parent'] is not None)
    if Counter((t['group'], tuple(t['source']), tuple(t['destination'])) for t in result['reduction_trees']) != expected_edges:
        raise ValueError('Original arithmetic reduction tree changed')
    for records in [f['routes'] for f in result['reduction_trees']] + [result['credit_fanouts']] + [f['routes'] for f in result['flows']]:
        for record in records:
            key = (*record['pe'], record['color'])
            if key in installed:
                raise ValueError('PE/color alias')
            if record['color'] not in (*COLORS, 18):
                raise ValueError('Reserved fabric color')
            x, y, w, h = region['rect']
            if not (x <= key[0] < x+w and y <= key[1] < y+h):
                raise ValueError('Route leaves original region')
            installed[key] = record
    def walk(source, color):
        frontier = [(tuple(source), color, 'RAMP')]; seen = set(); sinks = []
        while frontier:
            pe, c, incoming = frontier.pop(); key = (*pe, c)
            if key in seen:
                raise ValueError('Repeated physical channel/cycle')
            seen.add(key); route = installed[key]
            if route['rx'] != incoming:
                raise ValueError('Disconnected receive direction')
            directions = route['tx'] if isinstance(route['tx'], list) else [route['tx']]
            for tx in directions:
                if tx == 'RAMP':
                    sinks.append(pe); continue
                dx, dy = DELTA[tx]
                swap = route.get('color_swap_x' if dx else 'color_swap_y', False)
                frontier.append(((pe[0]+dx, pe[1]+dy), c ^ int(swap), OPPOSITE[tx]))
        return sinks, len(seen)
    for tree in result['reduction_trees']:
        sinks, _ = walk(tree['source'], tree['routes'][0]['color'])
        if sinks != [tuple(tree['destination'])]:
            raise ValueError('Reduction endpoint changed')
    coverage = Counter()
    for group in groups:
        sinks, count = walk(group['root_pe'], 18)
        expected = {tuple(rank_xy(region, group['rank_start']+i)) for i in range(group['workers']) if i != group['root']}
        if set(sinks) != expected or len(sinks) != len(expected) or count != group['workers']:
            raise ValueError('Credit fanout does not reach exactly its original workers')
    records_by_pair = {}
    for flow in result['flows']:
        root = groups[flow['producer']]['root_pe']; actor = result['actors'][flow['actor']]['pe']
        wanted = (root, actor) if flow['kind'] == 'projection' else (actor, root)
        if (flow['source'], flow['destination']) != wanted:
            raise ValueError('Stream not bound to original producer and assigned actor')
        sinks, count = walk(flow['source'], flow['routes'][0]['color'])
        if sinks != [tuple(flow['destination'])] or count != len(flow['routes']):
            raise ValueError('Fused endpoint disconnected')
        key = (flow['producer'], flow['actor'], flow['kind'])
        if key in records_by_pair:
            raise ValueError('Duplicate endpoint stream')
        records_by_pair[key] = flow
        stats['flows'] += 1; stats['route_entries'] += count
        stats['maximum_hops'] = max(stats['maximum_hops'], count-1)
        stats['color_swaps'] += sum(r.get('color_swap_x', False) + r.get('color_swap_y', False) for r in flow['routes'])
        if flow['kind'] == 'projection':
            for iteration in flow['iterations']:
                coverage[flow['producer'], iteration] += 1
                group = groups[flow['producer']]
                row = (group['output_start']+iteration)*group['rows']
                if row//128 not in result['actors'][flow['actor']]['groups']:
                    raise ValueError('Projected block sent to wrong original quantization group')
    for key, flow in records_by_pair.items():
        if key[2] == 'projection':
            credit = records_by_pair[key[:2]+('credit',)]
            if credit['source'] != flow['destination'] or credit['destination'] != flow['source'] or credit['iterations'] != flow['iterations']:
                raise ValueError('Copied-block credit is not bound to its actual producer')
    expected = {(g['index'], i) for g in groups for i in range(g['output_count'])}
    if set(coverage) != expected or set(coverage.values()) != {1}:
        raise ValueError('Missing or repeated original projected rows')
    return dict(passed=True, **stats, original_projected_blocks=len(expected),
                exact_worker_credit_fanout=True, single_input_per_route=True,
                numerical_qualified=False, complete_layer_connected=False)


def translate_routes(template, original_region, region):
    """Reuse routing search only for an identical relative resident geometry."""
    signature = lambda r: (r['rect'][2:], [m['shape'] for m in r['matrices']],
                           [m['group_partitions'] for m in r['matrices']],
                           [(a['rank'], a['resident_payload_bytes']) for a in r['quantization_actors']])
    if signature(original_region) != signature(region):
        raise ValueError('Route template differs from original resident geometry')
    dx, dy = (b-a for a, b in zip(original_region['rect'][:2], region['rect'][:2]))
    result = deepcopy(template)
    move = lambda pe: [pe[0]+dx, pe[1]+dy]
    for a in result['actors']: a['pe'] = move(a['pe'])
    for flow in result['reduction_trees'] + result['flows']:
        flow['source'] = move(flow['source']); flow['destination'] = move(flow['destination'])
        for route in flow['routes']: route['pe'] = move(route['pe'])
    for route in result['credit_fanouts']: route['pe'] = move(route['pe'])
    result.update(region=region['id'], rect=region['rect'])
    result['audit'] = audit_fused_routes(region, result)
    return result


def root_parameters(result, group):
    """Concrete two-port runtime parameters, matching statically bound credits."""
    streams = {(f['actor'], f['kind']): f for f in result['flows'] if f['producer'] == group['index']}
    main = streams[group['index'], 'projection']; credit = streams[group['index'], 'credit']
    params = dict(fused_root=True, out_color=main['routes'][0]['color'], main_credit_color=credit['routes'][-1]['color'])
    boundary = [f for f in streams.values() if f['kind'] == 'projection' and f['actor'] != group['index']]
    if boundary:
        b = boundary[0]; c = streams[b['actor'], 'credit']
        if len(boundary) != 1 or b['iterations'] != list(range(len(b['iterations']))):
            raise ValueError('Root needs more than one contiguous prefix stream')
        params.update(boundary_rows=len(b['iterations']), boundary_out_color=b['routes'][0]['color'],
                      boundary_credit_color=c['routes'][-1]['color'])
    else: params['boundary_rows'] = 0
    return params


def csl_routes(result, origin=(0, 0)):
    """Emit the checked fabric literally; callbacks and other operators are separate."""
    output = []
    for records in [f['routes'] for f in result['reduction_trees']] + [result['credit_fanouts']] + [f['routes'] for f in result['flows']]:
        for r in records:
            tx = r['tx'] if isinstance(r['tx'], list) else [r['tx']]
            fields = '.rx=.{%s},.tx=.{%s}' % (r['rx'], ','.join(tx))
            for flag in ['color_swap_x', 'color_swap_y']:
                if r.get(flag): fields += ',.%s=true' % flag
            output.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{%s}});' %
                          (r['pe'][0]-origin[0], r['pe'][1]-origin[1], r['color'], fields))
    return '\n'.join(output)+'\n'
