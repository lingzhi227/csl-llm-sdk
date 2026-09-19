"""Decode independent sink records; never authorize another device operation."""

EVENTS = {
    1: 'prepared', 2: 'compute_entered', 3: 'packet_entered', 4: 'packet_returned',
    5: 'QK_entered', 6: 'QK_returned', 7: 'attention_entered', 8: 'attention_returned',
    9: 'READY_queued', 10: 'READY_source_completed', 11: 'owner_finished',
}


def pair(value):
    return [value & 65535, value >> 16]


def bf16_summary(value):
    return dict(nonfinite_count=value & 65535, finite_absmax_BF16_bits=value >> 16)


def decode_head(values, head):
    require(len(values) == 32 and all(type(x) is int and 0 <= x <= 0xffffffff for x in values),
        'Exact unsigned head snapshot words')
    coherent = (values[0] == values[31] and 0 < values[0] <= 128 and values[0] % 2 == 0
        and values[1:4] == [0x48445031, 1 | (head << 16), 1]
        and values[5] == values[0]//2)
    result = dict(head=head, sink=[748, 24-head], raw_u32=values, coherent=coherent,
        full_layer_complete=False)
    if not coherent:
        return result
    q = pair(values[8])+pair(values[9]); k = pair(values[10]); v = pair(values[11])
    valid = values[25] & 3; seen = values[30] & 0xfff
    domain_counts = [(values[23] >> shift) & 255 for shift in (0, 8, 16, 24)]
    event_names = [EVENTS[i] for i in EVENTS if seen & (1 << i)]
    result.update(snapshot_sequence=values[0]//2, event=values[4], event_name=EVENTS.get(values[4], 'unknown'),
        event_count=values[5], received_packet_count=values[6],
        last_kind=values[7] & 255, last_group=(values[7] >> 8) & 255,
        last_offset=(values[7] >> 16) & 255, last_packet_length=values[7] >> 24,
        Q_counts=q, K_counts=k, V_counts=v, collector_count_unit='packed_u16_pairs',
        all_collectors_complete=all(x == 64 for x in q+k+v),
        attention_flags=values[12], QK_error=None if values[13] == 0xffffffff else values[13],
        attention_error=None if values[14] == 0xffffffff else values[14],
        wrapper_flags=values[15] & 65535, tx_mode=values[15] >> 16,
        attention_position=values[16] & 65535, served=values[16] >> 16,
        token_position=values[17], QK_event_nonzero_mask=values[18] & 127,
        attention_event_nonzero_mask=(values[18] >> 8) & 4095,
        summary_valid_bits=valid, packet_descriptor_overflow=bool(values[25] & (1 << 16)),
        wrapper_state0=values[26], QK_control0=values[27],
        QK_completion_marker=values[28], attention_completion_marker=values[29],
        event_seen_mask=seen, events_seen=event_names, sink_protocol_error=bool(values[30] & (1 << 30)),
        source_cap_reached=bool(values[30] & (1 << 31)), within_normal_event_bound=values[5] <= 61)
    # A zero summary before its validity bit is set is not a finite-data claim.
    result['before_QK'] = None if not valid & 1 else dict(
        QK_input=bf16_summary(values[19]), gains=bf16_summary(values[20]),
        V_nonfinite_count=(values[25] >> 17) & 511,
        gate_nonfinite_count=values[21] & 65535, gate_finite_domain_count=values[21] >> 16,
        gate_finite_absmax_BF16_bits=values[22] & 65535, gate_argmax=values[22] >> 16,
        frequency_nonfinite_count=domain_counts[0], frequency_finite_domain_count=domain_counts[1],
        probe_nonfinite_count=domain_counts[2], probe_finite_domain_count=domain_counts[3],
        original_gate_exp_argument='-abs(g)', original_gate_exp_domain=[-24, 0])
    result['after_QK'] = None if not valid & 2 else bf16_summary(values[24])
    result['field_bounds_valid'] = (
        values[4] in EVENTS and seen & (1 << values[4]) != 0 and values[6] <= 26
        and max(q+k+v) <= 64 and values[12] < 32
        and (values[15] & 65535) < 64 and values[15] >> 16 <= 3
        and values[16] & 65535 <= 1 and values[16] >> 16 <= 2 and values[17] == 0
        and values[18] & ~0xfff7f == 0 and values[25] & ~0x3ff0003 == 0
        and values[26] in (1, 2) and values[27] == 0
        and values[28] in (0, 7) and values[29] in (0, 7)
        and values[30] & ~0xc0000ffe == 0
        and (not valid & 2 or valid & 1)
        and bool(valid & 1) == bool(seen & (1 << 5))
        and bool(valid & 2) == bool(seen & (1 << 6))
        and bool(values[30] & (1 << 31)) == (values[5] == 64)
        and (not valid & 1 or (
            (values[19] & 65535) <= 512 and values[19] >> 16 < 0x7f80
            and (values[20] & 65535) <= 512 and values[20] >> 16 < 0x7f80
            and (values[25] >> 17) & 511 <= 256
            and (values[21] & 65535)+(values[21] >> 16) <= 256
            and (values[22] & 65535) < 0x7f80 and values[22] >> 16 < 256
            and sum(domain_counts[:2]) <= 32 and sum(domain_counts[2:]) <= 8))
        and (not valid & 2 or ((values[24] & 65535) <= 512 and values[24] >> 16 < 0x7f80)))
    # This describes the last coherent publication, not a proof of where execution stopped.
    result['last_record_stage'] = next((name for bit, name in (
        (11, 'owner_finished'), (10, 'READY_source_completed'), (9, 'READY_queued'),
        (8, 'attention_returned'), (7, 'attention_entered_without_return'),
        (6, 'QK_returned'), (5, 'QK_entered_without_return')) if seen & (1 << bit)),
        'collecting_QKV' if seen & (1 << 2) else 'prepared')
    return result


def decode_observation(rows):
    require(len(rows) == 25 and all(len(row) == 32 for row in rows), 'Exactly 25 independent observer rows')
    require(all(type(x) is int and 0 <= x <= 0xffffffff for row in rows for x in row),
        'Exactly 800 unsigned snapshot words')
    origin = decode_global(rows[0])
    origin['field_bounds_valid'] = False
    if origin['coherent']:
        try:
            semantic_state(origin)
            origin['field_bounds_valid'] = True
        except ValueError as error:
            origin['field_bounds_error'] = str(error)
    heads = [decode_head(rows[24-head], head) for head in range(24)]
    return dict(global_observer=origin, heads=heads,
        all_rows_coherent=origin['coherent'] and all(h['coherent'] for h in heads),
        independently_coherent_rows=True, globally_simultaneous=False,
        snapshot_rectangle=[748, 0, 1, 25], raw_words=800,
        full_layer_complete=False, full_math_acceptance=False)

def require(test, message):
    if not test:
        raise ValueError(message)


def decode_global(raw):
    values=[int(x) for x in raw]
    coherent=(values[0]==values[31] and 0<values[0]<=1922 and values[0]%2==0
              and values[1]==0x4f425331 and values[2]==1 and values[3]==1)
    result=dict(raw_u32=values, coherent=coherent, final_origin_snapshot=False)
    if not coherent:return result
    result.update(snapshot_sequence=values[0]//2, original_state=values[4:7], phase=values[7],
        offset=values[8], input_words=values[9], frame_flags=values[10], wrapper_flags=values[11],
        ready_counts=values[12:17], query_ready_mask=values[17], mlp_ready_masks=values[18:23],
        norm_ready_mask=values[23], frame_counters=values[24:29], request_phase=values[29],
        request_group=values[30]&65535, requested_offset=values[30]>>16)
    # Receiving may still be true during the callback that completed the origin;
    # every other ownership flag must be retired before even attempting the barrier.
    result['final_origin_snapshot']=(values[4:7]==[2,1,4] and values[7]==4
        and values[8]==17472 and values[9]==17408 and values[10]==32 and (values[11]&~8)==2
        and values[12:17]==[1,24,1,136,1] and values[17]==(1<<24)-1
        and values[18:23]==[(1<<32)-1]*4+[255] and values[23]==7
        and values[25:29]==[4,264,4*17472,0])
    return result


def semantic_state(observation):
    require(observation['coherent'], 'Coherent global observation required')
    counts=observation['ready_counts']; masks=[observation['query_ready_mask'], *observation['mlp_ready_masks'], observation['norm_ready_mask']]
    require(len(counts)==5 and all(0<=v<=n for v,n in zip(counts,[1,24,1,136,1])), 'READY count bounds')
    require(masks[0] < 1<<24 and all(0<=v<1<<32 for v in masks[1:5])
        and masks[5]<256 and masks[6]<8, 'READY mask bounds')
    require(masks[0].bit_count()==counts[1] and sum(v.bit_count() for v in masks[1:6])==counts[3]
        and [(masks[6]>>i)&1 for i in range(3)]==[counts[0],counts[2],counts[4]], 'Unique READY count/mask consistency')
    phase=observation['phase']; counters=observation['frame_counters']
    require(0<=phase<=4 and len(counters)==5 and 0<=counters[0]<=3
        and 0<=counters[1]<=4 and 0<=counters[2]<=264 and 0<=counters[3]<=69888
        and counters[4]==0 and observation['request_phase']<4
        and observation['request_group']<136 and observation['requested_offset']<5120,
        'Finite phase/frame/request bounds')
    # Request identity is lexicographically monotone across the four phases;
    # offsets may reset between groups and are not compared componentwise.
    return dict(cumulative=[phase,*counts,*counters[:4]], masks=masks,
        request=[observation['request_phase'],observation['request_group'],observation['requested_offset']])
