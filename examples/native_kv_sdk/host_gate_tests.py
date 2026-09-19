"""Synthetic capture-gate tests only; never imports or executes the SDK."""
from copy import deepcopy
import json
import time
from check_capture import PLACEMENTS, check_epoch, check_negative, packet_counts, pattern


def fixture(epoch):
    evidence = [[0] * 128 for _ in range(45)]
    slow, slot = {1: (5, 0), 2: (0, 3), 3: (3, 1)}[epoch]
    for (x, y), (role, index) in PLACEMENTS.items():
        row = evidence[y * 9 + x]
        row[1:9] = [role, index, epoch, 12 if epoch == 3 else 11, 1 if epoch == 2 else 0, 0, 6, 1]
        for k in range(11):
            row[96 + 2 * k] = 100 + 10 * k
        row[114] = 1000
        row[116] = 105
        tx, rx = packet_counts(role, index, epoch)
        for k in (0, 2, 3, 8):
            row[64 + k] = tx
        row[73] = rx
        row[78] = rx + 1
        if role in (1, 4):
            mask, count = (15, 4) if role == 1 else (1, 1)
            row[16:20] = [mask, 0, mask, count]
            row[21] = 1
        if role == 1:
            row[9] = row[37] = 1
            row[20] = 15
            row[24:29] = [4, 4, 4, 15, 0]
            order = [i for i in range(4) if i != slot] + [slot] if index == slow else [0, 1, 2, 3]
            row[30] = sum(v << (2 * i) for i, v in enumerate(order))
            row[31:33] = [4, 8 * epoch]
            if index == slow:
                row[35] = row[36] = row[38] = 1
        elif role == 2:
            row[24:32] = [24, 24, 63, 6, 0, 24, 0, 0]
            row[40] = int(epoch == 2)
        elif role == 3:
            row[24:30] = [1, 1, 1, 1, 0, 1]
            row[35] = 1
        elif role == 4:
            row[35] = 1
            row[36] = row[40] = int(epoch == 2 and index == 3)
        elif role == 5:
            row[32:35] = [8 * epoch, int(epoch < 3), 0]
    qk, inputs = [], []
    for head in range(6):
        q = [pattern(4 * head + i // 128, i % 128, epoch) for i in range(512)]
        k = [pattern(96 + i // 128, i % 128, epoch) for i in range(256)]
        v = [pattern(98 + i // 128, i % 128, epoch) for i in range(256)]
        qk.append([0x5aa5] * 2 + q[:256] + k + [0x5aa5] * 2)
        inputs.append([0x5aa5] * 2 + q + v + q[256:] + [0x5aa5] * 2)
    sources = [[0x5aa5] * 2 + [pattern(96 + s, i, epoch) for i in range(128)] + [0x5aa5] * 2 for s in range(4)]
    return [evidence, qk, inputs, sources, epoch]


def main():
    start = time.monotonic()
    fixtures = {epoch: fixture(epoch) for epoch in (1, 2, 3)}
    for args in fixtures.values():
        check_epoch(*args)
    negatives = deepcopy(fixtures[3][0])
    for (x, y), (role, index) in PLACEMENTS.items():
        row = negatives[y * 9 + x]
        row[48:52] = [14 if role in (1, 4) else 11, 93, 6, 1]
        if role in (1, 4):
            mask = 15 if role == 1 else 1
            row[52:57] = [13, 3, 0, mask, mask]
    check_negative(negatives, fixtures[3][0])
    faults = [
        ('QK payload', 1, lambda a: a[1][0].__setitem__(17, a[1][0][17] ^ 1)),
        ('Input canary', 1, lambda a: a[2][1].__setitem__(0, 0)),
        ('Native source payload', 1, lambda a: a[3][2].__setitem__(17, 0)),
        ('Missing native callback', 1, lambda a: a[0][0].__setitem__(19, 3)),
        ('Unreleased native lease', 1, lambda a: a[0][0].__setitem__(17, 1)),
        ('Actual TX/RX overlap', 2, lambda a: a[0][25].__setitem__(36, 0)),
        ('Q preservation/native overlap', 1, lambda a: a[0][5].__setitem__(35, 0)),
        ('Held stream order', 1, lambda a: a[0][5].__setitem__(30, 228)),
        ('Missing ACK source callback', 1, lambda a: a[0][0].__setitem__(26, 3)),
        ('RX not rearmed', 1, lambda a: a[0][0].__setitem__(78, 12)),
        ('Missing packet source callback', 1, lambda a: a[0][0].__setitem__(67, 4)),
        ('Consumer twice', 1, lambda a: a[0][0].__setitem__(9, 2)),
        ('Observer still live on final epoch', 3, lambda a: a[0][36].__setitem__(33, 1)),
        ('Wrong reset', 3, lambda a: a[0][0].__setitem__(4, 11)),
        ('Consumer before native completion', 1, lambda a: a[0][0].__setitem__(110, 0)),
        ('Non-halfword native payload', 1, lambda a: a[3][0].__setitem__(4, 65536)),
        ('Missing head row', 1, lambda a: a[1].pop()),
        ('Native order high bits', 1, lambda a: a[0][0].__setitem__(30, a[0][0][30] | 256)),
        ('Native send before compute entry', 1, lambda a: a[0][15].__setitem__(116, 999)),
    ]
    rejected = []
    for label, epoch, mutate in faults:
        args = deepcopy(fixtures[epoch]); mutate(args)
        try:
            check_epoch(*args)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Fault escaped: ' + label)
    for label, field, value in [('Negative count', 48, 13), ('Negative API changed live lease', 54, 1), ('Negative altered normal evidence', 41, 1)]:
        damaged = deepcopy(negatives); damaged[0][field] = value
        try:
            check_negative(damaged, fixtures[3][0])
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Negative fault escaped: ' + label)
    return dict(status='passed', synthetic_positive_epochs=3, synthetic_positive_negative_summary=1,
                deliberate_faults_rejected=rejected, seconds=time.monotonic() - start,
                SDK_imported_or_executed=False, CSL_compiled=False,
                scope='Host acceptance checker only; constructed arrays are not device evidence.')


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
