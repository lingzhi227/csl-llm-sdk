"""Decode retained original observations and six finite raw trace prefixes."""
from observer_decode import decode_observation as decode_original

SOURCES = (
    dict(identity=0, name='K1', x=71, y=1, row=1, maximum=62, packets=18, fanout=6, kind='root'),
    dict(identity=1, name='Q12', x=401, y=2, row=2, maximum=12, packets=3, fanout=1, kind='root'),
    dict(identity=2, name='Q64', x=401, y=6, row=6, maximum=12, packets=3, fanout=1, kind='root'),
    dict(identity=3, name='head0', x=16, y=0, row=24, maximum=106, packets=26, kind='head'),
    dict(identity=4, name='head3', x=19, y=0, row=21, maximum=106, packets=26, kind='head'),
    dict(identity=5, name='head16', x=32, y=0, row=8, maximum=106, packets=26, kind='head'),
)
NAMES = {1:'COMPUTE',2:'OUTPUT',3:'SDK_ENTER',4:'SDK_RETURN',5:'SOURCE_COMPLETE',
         6:'FANOUT_ADVANCE',7:'INITIAL_HEADER_ARM',8:'PAYLOAD_ARM',9:'CALLBACK_ENTER',
         10:'EVENT4_RETURN',11:'REARM'}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def fields(word):
    return dict(event=word >> 28, ordinal=(word >> 20) & 255,
                packet=(word >> 14) & 63, detail=(word >> 8) & 63, flags=word & 255)


def validate_events(tokens, source):
    compute = output = initial = False
    packet = fanout = 0
    waiting = None
    length = 0
    lengths = {31:0, 26:0, 8:0}
    decoded = []
    for index, word in enumerate(tokens, 1):
        f = fields(word); event = f['event']; p = f['packet']; detail = f['detail']; flags = f['flags']
        require(f['ordinal'] == index and flags & 248 == 0, 'emission ordinal or reserved flags')
        require(event in NAMES, 'reserved event')
        if event == 1:
            require(not compute and p == packet and detail == 0, 'COMPUTE count/current packet/detail')
            require(source['kind'] == 'root' or initial, 'head COMPUTE before initial header')
            compute = True
        elif source['kind'] == 'root':
            if event == 2:
                require(not output and packet == 0 and p == 0 and detail == 0, 'root OUTPUT point')
                output = True; waiting = 3
            elif event == 3:
                require(output and waiting == 3 and p == packet+1 and p <= source['packets'], 'root SDK_ENTER order')
                packet = p; length = (31,31,26)[(packet-1) % 3]
                require(detail == length and flags & 3 == 1, 'root SDK_ENTER length/lease')
                waiting = 4
            elif event == 4:
                require(waiting == 4 and p == packet and detail == length and flags & 3 == 1, 'root SDK_RETURN order/lease')
                waiting = 5
            elif event == 5:
                require(waiting == 5 and p == packet and detail == length and flags & 1 == 1, 'root completion before source lease clear')
                waiting = 6 if packet % 3 == 0 else 3
            elif event == 6:
                require(waiting == 6 and p == packet and detail == fanout+1 and flags & 1 == 0, 'root fanout after source release')
                fanout += 1; require(fanout <= source['fanout'], 'root fanout bound')
                waiting = 3
            else:
                raise ValueError('head event on root')
        else:
            if event == 7:
                require(not initial and index == 1 and p == 0 and detail == 0 and flags & 6 == 4, 'initial header order/flags')
                initial = True; waiting = 8
            elif event == 8:
                require(initial and waiting == 8 and p == packet+1 and p <= 26, 'payload arm order')
                require(detail in lengths and flags & 6 == 2, 'payload arm length/lease')
                packet = p; length = detail; lengths[length] += 1
                require(lengths[31] <= 16 and lengths[26] <= 8 and lengths[8] <= 2, 'head original packet budget')
                waiting = 9
            elif event in (9,10):
                require(waiting == event and p == packet and detail == length and flags & 6 == 2, 'head callback/event4 return order/lease')
                waiting = event+1
            elif event == 11:
                require(waiting == 11 and p == packet and detail == 0 and flags & 6 == 4, 'head rearm after application lease clear')
                waiting = 8
            else:
                raise ValueError('root event on head')
        decoded.append(dict(f, name=NAMES[event], raw_u32=word))
    return dict(events=decoded, compute_seen=compute, packet_ordinal=packet,
                fanout_advanced=fanout if source['kind']=='root' else None,
                last_event=decoded[-1]['name'] if decoded else None,
                next_protocol_event=NAMES.get(waiting), packet_length_counts=lengths if source['kind']=='head' else None)


def decode_trace(raw, source):
    result = dict(source=source['name'], source_id=source['identity'], source_xy=[source['x'],source['y']],
                  sink_xy=[749,source['row']], raw_u32=list(raw), valid=False)
    try:
        require(type(raw) is list and len(raw)==160 and all(type(v) is int and 0<=v<=0xffffffff for v in raw), 'real160-u32 trace extent')
        count=raw[7]
        require(count <= source['maximum'] and raw[0]==raw[31]==2*count, 'trace count and coherent endpoints')
        require(raw[1:7]==[0x46494631,1,source['identity'],source['x'],source['y'],source['maximum']], 'trace identity/version/maximum')
        require(raw[8:31]==[0]*23 and raw[32+count:]==[0]*(128-count), 'trace errors/reserved/padding')
        result.update(validate_events(raw[32:32+count],source), count=count, valid=True,
            reached_maximum_count=count==source['maximum'], observed_prefix_only=True,
            source_counters_observed=False, later_bound_violation_excluded=False,
            missing_event_unique_cause=False)
    except ValueError as error:
        result['error']=str(error)
    return result


def decode_observation(rows):
    require(type(rows) is list and len(rows)==50 and all(type(row) is list and len(row)==160 and
        all(type(v) is int and 0<=v<=0xffffffff for v in row) for row in rows), 'exact50x160-u32 idle rectangle')
    original=decode_original([rows[2*y][:32] for y in range(25)])
    original_padding_bad=[y for y in range(25) if rows[2*y][32:] != [0]*128]
    active_rows={s['row'] for s in SOURCES}
    inactive_bad=[y for y in range(25) if y not in active_rows and rows[2*y+1] != [0]*160]
    traces=[decode_trace(rows[2*s['row']+1],s) for s in SOURCES]
    original.update(snapshot_rectangle=[748,0,2,25],raw_words=8000,
        original_padding_bad_rows=original_padding_bad,inactive_trace_bad_rows=inactive_bad,
        traces=traces,all_trace_records_valid=all(t['valid'] for t in traces),
        all_padding_valid=not original_padding_bad and not inactive_bad,
        full_layer_complete=False,full_math_acceptance=False,globally_simultaneous=False,
        source_counters_observed=False,SDK_output_stall_qualified=False)
    return original
