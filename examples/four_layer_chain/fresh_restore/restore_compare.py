"""Compare complete actual semantic banks after independently released runs.

This is a pure evidence reader. The caller must independently bind the two
capture files, successful physical lifecycles and actual release receipts.
"""
from pathlib import Path
import hashlib
from checkpoint import _payload, identity, control
from runtime_boundary import require


def compare(baseline_root, baseline, restored_root, restored, host, lowered):
    import numpy as np
    require(baseline['mode'] == 'baseline' and baseline['completed'] == [1, 2, 3] and
            restored['mode'] == 'restore' and restored['completed'] == [1] and
            baseline['normal_stop'] is True and restored['normal_stop'] is True,
            'Both complete schedules and normal physical stop required')
    require(baseline['artifact_sha256'] == restored['artifact_sha256'] and
            baseline['prepared_pins'] == restored['prepared_pins'] and
            set(baseline['prepared_pins']) == {'0', '1', '2', '3'},
            'Identical actual artifact and own-layer original weights')
    require(baseline_root != restored_root, 'Independent physical runtime evidence directories')
    chosen = []
    for capture, serial in [(baseline, 2), (restored, 1)]:
        snapshots = [s for s in capture['semantic_snapshots'] if s['serial'] == serial and
                     s['generation'] == 1 and s['position'] == 1]
        require(len(snapshots) == 1, 'One exact completed logical-position1 snapshot')
        chosen.append(snapshots[0]['records'])
    expected = host['copies']['checkpoint_copies']
    require(len(chosen[0]) == len(chosen[1]) == len(expected) == 39, 'Every semantic array/control bank')
    rows, total = [], 0
    for item, left, right in zip(expected, *chosen):
        require(identity(item) == identity(left['copy']) == identity(right['copy']),
                'Exact full semantic geometry and dtype')
        a = _payload(Path(baseline_root), item, left['receipt'], False)
        b = _payload(Path(restored_root), item, right['receipt'], False)
        require(a.dtype == b.dtype and a.shape == b.shape and
                np.array_equal(a.view('<u2') if item['bits'] == 16 else a.view('<u4'),
                               b.view('<u2') if item['bits'] == 16 else b.view('<u4')),
                'Fresh restored position1 differs from uninterrupted position1: ' + str(identity(item)))
        if item['symbol'] == 'checkpoint_control':
            control(a, lowered['regions'], lowered['request_id'], lowered['stage_id'], 1, 2)
        total += a.nbytes
        rows.append(dict(symbol=item['symbol'], layer=item.get('layer_id'),
                         rectangle=[item[k] for k in ('x', 'y', 'width', 'height', 'count')],
                         native_bytes=a.nbytes, native_sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest(),
                         baseline=left['receipt']['path'], restored=right['receipt']['path'], exact=True))
    require(total == host['checkpoint_native_bytes'] == 14378752, 'Full hidden/DeltaNet/conv/KV/control extent')
    return dict(status='all_original_layer_semantic_banks_bit_exact_after_restored_position1',
                files=39, native_bytes=total, arrays=rows, checkpoint_position=0, compared_position=1,
                local_serials=[2, 1], transient_transport_compared=False, arithmetic_oracle=False,
                physical_release_verified_by_caller=True, full_model=False)
