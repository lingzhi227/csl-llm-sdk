"""Strict host gate for actual native/Q fixture readbacks; no model arithmetic."""
from pathlib import Path
import json

LAYOUT = json.loads(Path(__file__).with_name('layout.json').read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def words(value, rows, count, bits=32):
    require(type(value) is list and len(value) == rows, 'Readback row shape')
    for row in value:
        require(type(row) is list and len(row) == count, 'Readback word shape')
        require(all(type(x) is int and 0 <= x < 1 << bits for x in row), 'Readback integer domain')
    return value


def pattern(stream, index, epoch):
    if index < 4:
        return (0, 0x8000, 1, 0x8001)[index]
    return (0x3e00 + (stream * 131 + epoch * 37 + index) % 512) | (0x8000 if index % 2 else 0)


def placements():
    result = {(x, y): (0, 0) for y in range(5) for x in range(9)}
    result[8, 0] = 2, 0
    for role, label in ((1, 'heads'), (3, 'q_roots'), (4, 'kv_roots'), (5, 'observer_sinks')):
        for index, point in enumerate(LAYOUT[label]):
            require(result[tuple(point)] == (0, 0), 'Distinct fixture role placement')
            result[tuple(point)] = role, index
    return result


PLACEMENTS = placements()


def packet_counts(role, index, epoch):
    if role == 1:
        return 5 * epoch, 12 * epoch
    if role == 2:
        return 24 * epoch + (epoch >= 2), 30 * epoch
    if role == 3:
        return 4 * epoch, 2 * epoch
    if role == 4 and index == 3:
        return 0, int(epoch >= 2)
    return 0, 0


def cycle(row, slot):
    require(row[97 + 2 * slot] < 1 << 16, '48-bit timestamp domain')
    return row[96 + 2 * slot] | row[97 + 2 * slot] << 32


def check_epoch(evidence, qk, inputs, native_sources, epoch):
    require(epoch in (1, 2, 3), 'Three declared epochs')
    words(evidence, 45, 128)
    words(qk, 6, 516, 16)
    words(inputs, 6, 1028, 16)
    words(native_sources, 4, 132, 16)
    slow_head, slow_slot = {1: (5, 0), 2: (0, 3), 3: (3, 1)}[epoch]
    total_tx = total_rx = 0
    intervals = {}
    for (x, y), (role, index) in PLACEMENTS.items():
        row = evidence[y * 9 + x]
        require(row[0] == 0 and row[1:8] == [role, index, epoch, 12 if epoch == 3 else 11,
                1 if epoch == 2 else 0, 0, 6], 'Exact epoch identity and zero failure')
        require(row[8] == 1, 'Actual local command completion')
        require(cycle(row, 9) >= cycle(row, 0), 'Local prepare-to-completion timestamp')
        intervals[f'{x},{y}'] = cycle(row, 9) - cycle(row, 0)
        tx, rx = packet_counts(role, index, epoch)
        require(all(row[64 + i] == tx for i in (0, 2, 3, 8)), 'All queued packet sources completed')
        require(row[68] == row[71] == 0, 'No adapter lease/order violations')
        require(row[73] == rx and row[78] == rx + 1, 'Every RX payload returned and header rearmed')
        total_tx += tx
        total_rx += rx
        if role in (1, 4):
            mask, count = (15, 4) if role == 1 else (1, 1)
            require(row[16:20] == [mask, 0, mask, count] and row[21:24] == [1, 0, 0],
                    'All native streams released and retired without failure')
        if role == 1:
            require(row[9] == 1 and row[20] == 15 and row[24:29] == [4, 4, 4, 15, 0],
                    'Q receipt releases and native retention join once before consumer READY')
            require(row[31] == 4 and row[30] < 256 and sorted((row[30] >> (2 * i)) & 3 for i in range(4)) == [0, 1, 2, 3],
                    'Four distinct native completion callbacks')
            require(row[32] == 8 * epoch and row[37] == 1, 'Bounded observer and exactly one query preservation copy')
            require(row[29] <= 3, 'Finite pending receipt accounting')
            if index == slow_head:
                require(row[35] == 1 and row[36] == 1 and row[38] >= 1,
                        'Actual Q receive/copy overlapped a held native descriptor')
                require((row[30] >> 6) & 3 == slow_slot, 'Held native stream completed last')
            else:
                require(row[35] <= 1 and row[36] == 0, 'No undeclared consumer hold')
            require(cycle(row, 7) >= max(cycle(row, i) for i in (*range(2, 7), 10)), 'Consumer after all Q/native retention')
            require(cycle(row, 9) >= cycle(row, 8) >= cycle(row, 7), 'READY source completion before command retirement')
        elif role == 2:
            require(row[24:27] == [24, 24, 63] and 1 < row[27] <= 6 and row[28:32] == [0, 24, 0, 0],
                    'Concurrent Q window and all semantic/source leases retired')
            require(row[40] == int(epoch == 2), 'Actual raw source-start notice received')
        elif role == 3:
            require(row[24:30] == [1, 1, 1, 1, 0, 1] and row[35] == 1,
                    'Q root source/receipt join, immutable input and DONE source callback')
        elif role == 4:
            require(row[35] == 1, 'Native source payload and canaries preserved until source completion')
            held = int(epoch == 2 and index == 3)
            require(row[36] == row[40] == held, 'Actual blocked native TX plus simultaneous packet RX')
            require(cycle(row, 9) >= cycle(row, 4) >= cycle(row, 3) >= cycle(row, 10), 'Native source completion before command retirement')
        elif role == 5:
            require(row[32:35] == [8 * epoch, int(epoch < 3), 0], 'Observer final publication consumed; finite stream closes')
    expected_messages = 150 * epoch + int(epoch >= 2)
    require(total_tx == total_rx == expected_messages, 'Exact cumulative application message totals')
    for head in range(6):
        expected_qk = [pattern(4 * head + i // 128, i % 128, epoch) for i in range(256)]
        expected_qk += [pattern(96 + i // 128, i % 128, epoch) for i in range(256)]
        expected_input = [pattern(4 * head + i // 128, i % 128, epoch) for i in range(512)]
        expected_input += [pattern(98 + i // 128, i % 128, epoch) for i in range(256)]
        expected_input += [pattern(4 * head + 2 + i // 128, i % 128, epoch) for i in range(256)]
        require(qk[head] == [0x5aa5] * 2 + expected_qk + [0x5aa5] * 2, 'Actual Q/K retained bits and canaries')
        require(inputs[head] == [0x5aa5] * 2 + expected_input + [0x5aa5] * 2, 'Actual Q/raw-gate/V/preserved-gate bits and canaries')
    for stream in range(4):
        require(native_sources[stream] == [0x5aa5] * 2 + [pattern(96 + stream, i, epoch) for i in range(128)] + [0x5aa5] * 2,
                'Actual native source readback matches every receiver')
    return dict(status='passed', epoch=epoch, normal_quiescent_reset=epoch == 3,
                exact_BF16_values=6 * 1536 + 4 * 128, canary_halfwords=64,
                cumulative_application_messages=total_tx, local_prepare_to_completion_cycles=intervals,
                timing_scope='Local intervals include setup and observability; no cross-PE clock or performance comparison.')


def check_negative(evidence, previous):
    words(evidence, 45, 128)
    words(previous, 45, 128)
    total = 0
    for (x, y), (role, index) in PLACEMENTS.items():
        row = evidence[9 * y + x]
        before = previous[9 * y + x]
        require(row[:48] == before[:48] and row[57:] == before[57:], 'Negative phase modified undeclared normal evidence')
        count = 14 if role in (1, 4) else 11
        require(row[0] == 0 and row[48:52] == [count, 93, 6, 1], 'Actual shared native lifecycle rejections')
        if role in (1, 4):
            mask = 15 if role == 1 else 1
            require(row[52:57] == [13, 3, 0, mask, mask], 'Real role APIs rejected replay/arm/retire without a new DSD')
        else:
            require(row[52:57] == [0] * 5, 'No invented role rejection evidence')
        total += count
    require(total == 525, 'Exact negative branch count')
    return dict(status='passed', actual_rejected_calls=total, SDK_messages_sent=0,
                scope='Isolated state/API checks after three captured normal epochs, not a fourth neural epoch.')
