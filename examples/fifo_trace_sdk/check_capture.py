"""Exact finite histories and fixed host-order evidence; no neural claims."""
def require(value, message):
    if not value:
        raise ValueError(message)


def token(x, y, ordinal):
    return 0x80000000 | y << 24 | x << 16 | ordinal * 0x101


def return_token(y):
    return 0xf0000000 | y << 24 | (62 if y == 0 else 106)


def packet_token(x, y, ordinal):
    return 0x9a000000 | y << 20 | x << 16 | ordinal * 0x101


def rectangle(value, pes, words, label):
    require(type(value) is list and len(value) == pes, label + ' PE extent')
    require(all(type(row) is list and len(row) == words and
        all(type(v) is int and 0 <= v <= 0xffffffff for v in row)
        for row in value), label + ' actual uint32 extent')


def initial(captures):
    states = captures['initial']
    rectangle(states, 16, 160, 'initial')
    accepted_capacity = []
    for y, capacity, legal in ((0, 64, 62), (1, 128, 106)):
        backing = captures['fifo' + str(y)]
        rectangle(backing, 5, capacity, 'FIFO backing')
        for x in range(8):
            row = states[y * 8 + x]
            expected = [1, x, y, 1] + [0] * 156
            expected[15] = 1
            if x in (0, 3, 4):
                attempts = capacity + 1 if x == 3 else legal - (x == 0)
                outcomes = row[16:16 + attempts]
                accepted = sum(outcomes)
                require(outcomes == [1] * accepted + [0] * (attempts - accepted), 'FIFO success prefix')
                require(legal <= accepted <= capacity if x == 3 else accepted == attempts,
                        'Measured usable capacity / complete legal burst')
                if x == 3:
                    require(accepted < attempts, 'Capacity+1 must report full')
                    accepted_capacity.append(accepted)
                expected[4:8] = [attempts, accepted, attempts - accepted, 0]
                expected[16:16 + attempts] = outcomes
                require(backing[x] == [token(x, y, i + 1) for i in range(accepted)] +
                    [0] * (capacity - accepted), 'Actual FIFO data and zero padding')
            elif x < 5:
                require(backing[x] == [0] * capacity, 'Non-FIFO backing padding')
            require(row == expected, 'Exact initial evidence and reserved words')
    return accepted_capacity


def trace(capture, x, final):
    rectangle(capture, 2, 160, 'trace')
    for y, legal in ((0, 62), (1, 106)):
        count = legal if final or x == 4 else legal - 1
        words = [token(x, y, i + 1) for i in range(count)]
        if final and x == 0:
            words[-1] = return_token(y)
        expected = [0] * 160
        expected[0] = expected[31] = 2 * count
        expected[1:8] = [0x46494631, 1, x + 8 * y, x, y, legal, count]
        expected[32:32 + count] = words
        require(capture[y] == expected, 'Exact trace sequence, identity, history, error and padding')


def after_drain(captures):
    trace(captures['final_trace'], 0, True)
    trace(captures['budget_trace'], 4, True)


def final(captures):
    capacity = initial(captures)
    after_drain(captures)
    rectangle(captures['final'], 16, 160, 'final')
    rectangle(captures['payload'], 16, 31, 'payload')
    rectangle(captures['sideband'], 1, 32, 'sideband')
    for y, legal in ((0, 62), (1, 106)):
        for x in range(8):
            row = captures['final'][y * 8 + x]
            expected = list(captures['initial'][y * 8 + x])
            expected[0] = 4
            if x == 0:
                expected[4:8] = [legal, legal, 0, 0]
                expected[8:12] = [1, 0xdead3100 | y, 1, 1]
                expected[13] = y
            elif x == 1:
                expected[9] = 0xcafe3200 | y
                expected[12] = 1
            elif x == 2:
                expected[9] = legal - 1
                expected[12] = 1
            elif x == 7:
                expected[10:12] = [1, 1]
            elif x == 6 and y == 1:
                expected[13] = 1
            if x in (0, 7):
                expected[15] = 2
                stats = row[144:160]
                completed_at_header = stats[12]
                require(completed_at_header in (0, 1) if x == 0 else completed_at_header == 0,
                        'Observed TX completion prefix at RX header')
                fixed = [1, int(x == 7), 1, 1, 0, 0, 0, 0, 1, 1,
                         int(x == 0 and completed_at_header == 0), int(x == 0),
                         completed_at_header, int(x == 0), 2, 1]
                require(stats == fixed, 'Original packet lease, return and rearm diagnostics')
                expected[144:160] = stats
                require(captures['payload'][y * 8 + x] ==
                    [packet_token(7 if x == 0 else 0, y, i + 1) for i in range(31)],
                    'Original SDK packet payload bits')
            else:
                require(captures['payload'][y * 8 + x] == [0] * 31, 'Idle payload padding')
            require(row == expected, 'Exact final evidence and reserved words')
    expected_side = [0x11010000 | i for i in range(32)]
    expected_side[0] = expected_side[31] = 2
    require(captures['sideband'][0] == expected_side, 'Original synchronous sideband snapshot')
    return dict(complete_data_recovered=True, legal_bursts_qualified=True,
        measured_usable_capacity=capacity, main_task_drain_barrier_qualified=True,
        CPU_stalled_throughout_drain_qualified=False,
        independent_drain_qualified=True, packet_sideband_coexistence_qualified=True,
        idle_header_TX_qualified=True, header_rearm_qualified=True,
        SDK_output_stall_qualified=False, hardware=False, neural_math=False,
        copies=8, launches=3, host_slot_bytes=28992, H2D=0)
