"""Independent raw-u32 acceptance contract for static transport schema v1.

No SDK import, CSL execution, hardware claim or legacy650 reinterpretation.
The caller supplies exact raw exports after bounded device completion. Capture
ownership, stable reread, normal stop and source/artifact pins are outer gates.
"""
from collections import Counter

SCHEMA = 'static-ready-row-v1'


def ready(owner):
    return [10, 1, 1, 0, 0, 3, owner, 0]


def request(group):
    return [11, 1, 1, 0, 0, 2, group, group * 128]


def fragment(group, offset):
    n = {0: 23, 23: 23, 46: 18}[offset]
    body = []
    for i in range(n):
        low = 0x3f00 + group * 128 + 2 * (offset + i)
        body.append(low | ((low + 1) << 16))
    return [9, 1, 1, 0, 0, 2, group, offset] + body


def require(condition, label):
    if not condition:
        raise ValueError(label)


def words(value, n, label):
    require(isinstance(value, list) and len(value) == n, label + ': extent')
    require(all(type(x) is int and 0 <= x <= 0xffffffff for x in value), label + ': u32')
    return value


def rx_status(n, last=8):
    return [n, n, n, 0, last if n else 0, 0, 0, 0]


def tx_status(n, last=8):
    return [n, n, n, 0, last if n else 0, 0, n, n]


def decode_records(raw, capacity, expected, label):
    words(raw, capacity * 16, label)
    records = []
    for i in range(expected):
        r = raw[i * 16:(i + 1) * 16]
        require(r[0] == r[15] == 2 * (i + 1), label + ': torn/sequence')
        require(r[13] == 255 and r[14] == 0, label + ': mask/error')
        records.append(r)
    require(not any(raw[expected * 16:]), label + ': extra records')
    return records


def match_records(records, expected, label):
    """Compare independently specified (event, payload, route) sequences."""
    require(len(records) == len(expected), label + ': count')
    for r, (event, data, route) in zip(records, expected):
        subject = 0 if event == 5 else data[6]
        require(r[1:5] == [event, subject, len(data), route], label + ': metadata')
        require(r[5:13] == data[:8], label + ': identity')


def validate(capture):
    """Reject incomplete, extra, duplicate, corrupt or unreleased raw captures."""
    require(capture.get('schema') == SCHEMA, 'schema')
    first, count = capture.get('first'), capture.get('count')
    require(type(first) is int and type(count) is int, 'geometry types')
    require(first >= 4 and 1 <= count <= 136 and first + count + 2 <= 178, 'geometry')
    gate=capture.get('qualification_gate')
    require(type(gate) is bool and (not gate or (first,count)==(4,3)), 'qualification gate scope')
    width, last = first + count + 2, first + count - 1
    tiles = capture.get('tiles', {})
    require(set(tiles) == {f'{x},{y}' for y in range(3) for x in range(width)}, 'tile coverage')
    sink_records, relay_records = {}, {}
    for y in range(3):
        for x in range(width):
            t = tiles[f'{x},{y}']; label = f'tile {x},{y}'
            expected_exports={'state','input_a_status','input_b_status','output_status'}
            expected_exports.update({'relay_records'} if y==2 else {'diagnostic_status','diagnostic_records'})
            if (x,y)==(0,0):expected_exports.add('row_packets')
            if gate and (x,y)==(first,2):expected_exports.add('gate_words')
            require(set(t) == expected_exports, label + ': captured exports')
            state = words(t['state'], 32, label + ': state')
            producer = first <= x <= last
            role = 5 if y == 2 else 4 if y == 1 else 1 if x == 0 else 2 if x == 2 else 3 if producer else 0
            require(state[:5] == [2, role, 1, x, 0], label + ': completion/error')
            for key in ('input_a_status', 'input_b_status', 'output_status'):
                words(t[key], 8, label + ':' + key)
            if y == 2:
                local = int(producer)
                upstream = sum(first + owner > x for owner in range(count))
                total = local + upstream
                require(state[5:11] == [local, upstream, total, local, upstream, total], label + ': conservation')
                require(state[11] == state[12] == 0 and state[13] + state[14] == total, label + ': arbitration')
                if gate and x==first:
                    selected=state[17]
                    require(selected in (1,2), label + ': gate selected')
                    require(state[16:23] == [1,selected,3 if selected==1 else 2,3 if selected==2 else 2,1,0,2], label + ': held-callback witness')
                else:
                    require(not any(state[16:23]), label + ': unexpected qualification gate')
                require(state[15]==0 and not any(state[23:]), label + ': reserved')
                require(t['input_a_status'] == rx_status(local), label + ': local lease')
                require(t['input_b_status'] == rx_status(upstream), label + ': upstream lease')
                require(t['output_status'] == tx_status(total), label + ': output lease')
                raw = words(t['relay_records'], count * 16, label + ': relay records')
                rs = {}
                for owner in range(count):
                    r = raw[owner * 16:(owner + 1) * 16]
                    if first + owner < x:
                        require(not any(r), label + ': unexpected owner')
                        continue
                    port = 1 if first + owner == x else 2
                    require(r[0] == r[15] == 6 and r[1:3] == [port, owner], label + ': record release/provenance')
                    require(r[5:13] == ready(owner) and r[13:15] == [3, 0], label + ': full READY/point checks')
                    rs[owner] = r
                require(sorted(r[4] for r in rs.values()) == list(range(1, total + 1)), label + ': output order')
                for port, n in ((1, local), (2, upstream)):
                    require(sorted(r[3] for r in rs.values() if r[1] == port) == list(range(1, n + 1)), label + ': ingress order')
                if gate and x==first:
                    witness=words(t['gate_words'],16,label+': full gate inputs')
                    other=witness[14]
                    require(witness[:8]==ready(0) and 0<other<count and witness[8:]==ready(other),label+': gate payload provenance')
                    selected_owner=0 if state[17]==1 else other
                    require(rs[selected_owner][4]==1,label+': gate first source')
                relay_records[x] = rs
                continue
            origin, peer = y == 0 and x == 0, y == 0 and x == 2
            source = y == 0 and producer
            require(t['input_a_status'] == rx_status(count if origin else 8 if peer else 0), label + ': metadata receive')
            require(t['input_b_status'] == rx_status(24 if origin else 0, 26), label + ': row receive')
            require(t['output_status'] == tx_status(8 if origin else 24 if peer else 1 if source else 0, 26 if peer else 8), label + ': source release')
            if origin:
                require(state[8:16] == [count, 8, 0, 8, 0, 0, 0, 24], label + ': origin state')
                packets = words(t['row_packets'], 24 * 32, 'all row fragments')
                for k, (group, off) in enumerate((g, o) for g in range(8) for o in (0, 23, 46)):
                    expected = fragment(group, off)
                    require(packets[32*k:32*(k+1)] == [len(expected)] + expected + [0] * (31-len(expected)), 'full row payload/padding')
            else:
                if peer:
                    require(state[8:16] == [0, 0, 8, 0, 0, 8, 0, 0], label + ': pending request/peer state')
                elif source:
                    require(state[8:16] == [0, 0, 0, 0, 0, 0, 1, 0], label + ': producer state')
                else:
                    require(not any(state[8:16]), label + ': idle state')
            require(not any(state[5:8]) and not any(state[16:]), label + ': reserved state')
            enabled = x in (0, 2) or producer
            capacity = 192 if x == 0 else 96 if x == 2 else 3 if producer else 1
            expected_count = count + 49 if x == 0 else 81 if x == 2 else 3 if producer else 0
            status = words(t['diagnostic_status'], 8, label + ': diagnostics')
            require(status == [1, expected_count, expected_count, 0, 0, int(y == 1 and enabled), capacity, 1], label + ': diagnostics drain')
            records = decode_records(t['diagnostic_records'], capacity, expected_count, label)
            if y == 1:
                sink_records[x] = records
                require(t['diagnostic_records'] == tiles[f'{x},0']['diagnostic_records'], label + ': source/sink exact')
    # Across each directed link the right relay's output order is the left
    # relay's upstream receive order. Payload and exact owner coverage were
    # checked above, so count equality cannot hide replacement or duplication.
    for x in range(last):
        right = relay_records[x+1]
        for owner, record in right.items():
            require(relay_records[x][owner][1] == 2 and relay_records[x][owner][3] == record[4], f'link {x+1}->{x}: provenance/order')
    for owner in range(count):
        match_records(sink_records[first+owner], [(event, ready(owner), 0) for event in (1, 2, 3)], 'producer source points')
    origin = sink_records[0]
    ready_records = [r for r in origin if r[1] == 4 and r[4] == 9]
    order = sorted(relay_records[0], key=lambda owner: relay_records[0][owner][4])
    match_records(ready_records, [(4, ready(owner), 9) for owner in order], 'origin READY from endpoint')
    match_records([r for r in origin if r[1] in (1, 2, 3)], [(e, request(g), 0) for g in range(8) for e in (1, 2, 3)], 'origin REQUEST source points')
    match_records([r for r in origin if r[1] == 4 and r[4] == 13], [(4, fragment(g,o), 13) for g in range(8) for o in (0,23,46)], 'origin fragment headers')
    match_records([r for r in origin if r[1] == 5], [(5, [count,8,0,0,0,0,0,0], 0)], 'origin terminal')
    require(Counter(r[1] for r in origin) == {1:8,2:8,3:8,4:count+24,5:1}, 'origin unknown records')
    peer = sink_records[2]
    match_records([r for r in peer if r[1] == 4], [(4, request(g), 12) for g in range(8)], 'peer ordered REQUESTs')
    match_records([r for r in peer if r[1] in (1,2,3)], [(e,fragment(g,o),0) for g in range(8) for o in (0,23,46) for e in (1,2,3)], 'peer fragment source points')
    match_records([r for r in peer if r[1] == 5], [(5,[8,0,0,0,0,0,0,0],0)], 'peer terminal')
    require(Counter(r[1] for r in peer) == {1:24,2:24,3:24,4:8,5:1}, 'peer unknown records')
    require(origin[-1][1]==peer[-1][1]==5,'terminal must be last on each PE')
    # Same-PE diagnostic sequence numbers establish application causality.
    # No cross-PE timestamp or all-READY-before-REQUEST assumption is made.
    def ordinal(records,event,kind,group,offset=None):
        matches=[r[0]//2 for r in records if r[1]==event and r[5]==kind and r[11]==group and (offset is None or r[12]==offset)]
        require(len(matches)==1,'causal event uniqueness')
        return matches[0]
    for group in range(8):
        require(ordinal(origin,2,11,group)<ordinal(origin,4,9,group,0),'origin request issue before row arrival')
        require(ordinal(peer,4,11,group)<ordinal(peer,1,9,group,0),'peer REQUEST before row source acquire')
        if group:
            acquire=ordinal(origin,1,11,group)
            require(acquire>ordinal(origin,4,9,group-1,46),'origin next REQUEST after last fragment')
            require(acquire>ordinal(origin,3,11,group-1),'origin next REQUEST after source completion')
            require(ordinal(peer,1,9,group,0)>ordinal(peer,3,9,group-1,46),'peer next row after last source completion')
            require(ordinal(peer,4,11,group)>ordinal(peer,2,9,group-1,46),'peer next REQUEST after previous last issue')
    return dict(schema=SCHEMA, count=count, qualification_gate=gate,ready=count, requests=8, fragments=24,
                halfwords=1024, relay_links=last, neural_epochs=0,
                scope='raw schema validation only; execution provenance and normal-stop gates external')
