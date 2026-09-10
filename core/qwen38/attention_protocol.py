"""One-PE cache8 attention transport and state contract."""
ARRAYS = {'cache': (4098, 16), 'input': (1026, 16), 'scores': (42, 32),
          'score_casts': (26, 16), 'stats': (3, 32), 'stages': (1282, 32),
          'casts': (770, 16), 'state': (16, 32), 'events': (12, 32), 'control': (2, 32)}
READS = tuple(n for n in ARRAYS if n != 'control')
INITIAL = ('cache', 'scores', 'score_casts', 'stages', 'casts')
NORMAL_EVENTS = [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 1]
OVERFLOW_EVENTS = [1, 99, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
LEDGER = {'shape': [1, 1], 'head_dim': 256, 'cache_capacity': 8,
          'cache_payload_bytes': 8192, 'app_colors': [], 'app_async_tasks': [],
          'physical_host_buffer_limit': 65536, 'stack_allowance_bytes': 4096,
          'ordinary_address_ceiling': 49152, 'successful_tokens': 10,
          'overflow_calls': 1, 'initialize_calls': 2, 'launches': 13,
          'reset': 'metadata-only; full physical cache observed before/after',
          'numerical_work': 'device only; host inputs are original BF16 fixtures'}


def plan():
    rows = [('h2d', n) for n in INITIAL]
    for call in range(1, 11):
        if call in (1, 9):
            rows += [('h2d', 'control'), ('d2h', 'state'), ('d2h', 'cache')]
        rows += [('h2d', 'input'), ('h2d', 'control')] + [('d2h', n) for n in READS]
        if call == 8:
            rows += [('h2d', 'control')] + [('d2h', n) for n in READS]
    return rows


def budget():
    rows = plan()
    return dict(calls=len(rows), host_slots=sum(ARRAYS[n][0] for _, n in rows),
                payload_bytes=sum(ARRAYS[n][0]*ARRAYS[n][1]//8 for _, n in rows),
                max_host_buffer_bytes=max(ARRAYS[n][0]*4 for _, n in rows), launches=13)


def expected_state(generation, token, total, inits, rejects, initialized=False):
    return [generation, token, total, inits, token, 0 if initialized else 3, 0,
            rejects, 0 if initialized else 1, total, token, 8, 256, 0,
            0 if initialized else 0xa55a, 0 if initialized else 0x5aa5]


def validate_ledger():
    assert budget() == dict(calls=131, host_slots=104757, payload_bytes=242088,
                           max_host_buffer_bytes=16392, launches=13)
    assert 2*8*256*2 == LEDGER['cache_payload_bytes']
    assert budget()['max_host_buffer_bytes'] <= LEDGER['physical_host_buffer_limit']
    return True
