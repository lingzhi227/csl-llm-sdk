"""Finite complete baseline/restore copy schedules, without runtime admission."""


def build(copies, mode):
    if mode not in ('baseline', 'restore'):
        raise ValueError('One full baseline or one fresh restored position')
    baseline = mode == 'baseline'
    tokens = 3 if baseline else 1
    h2d = copies['uploads'] + tokens * [copies['hidden_input']]
    raw_h2d = tokens * [copies['hidden_input']]
    if not baseline:
        h2d += copies['checkpoint_copies']
        raw_h2d += copies['checkpoint_copies']
    raw_d2h = ((8 if baseline else 3) * copies['state_copies'] +
               (2 if baseline else 0) * copies['persistent_copies'] +
               2 * tokens * copies['transport_copies'] +
               tokens * copies['diagnostic_copies'] +
               tokens * 128 * copies['observers'] +
               tokens * copies['control_copies'] +
               tokens * copies['handoff_copies'] +
               (2 if baseline else 1) * copies['checkpoint_copies'])
    d2h = 2 * copies['uploads'] + raw_d2h
    def host(p):
        return p['width'] * p['height'] * p['count'] * 4
    def native(p):
        return host(p) * p['bits'] // 32
    hbytes, dbytes = sum(map(host, h2d)), sum(map(host, d2h))
    rbytes = sum(map(native, raw_h2d + raw_d2h))
    rfiles = len(raw_h2d + raw_d2h)
    return dict(mode=mode, max_copies=len(h2d+d2h), max_H2D_host_bytes=hbytes,
                max_D2H_host_bytes=dbytes, max_host_bytes=hbytes+dbytes,
                expected_raw_bytes=rbytes, expected_raw_files=rfiles,
                failed_copy_raw_reserve_bytes=16 << 20, raw_byte_ceiling=rbytes+(16 << 20),
                raw_file_ceiling=rfiles+1, maximum_copy_host_bytes=max(map(host, h2d+d2h)),
                max_launches=8 if baseline else 3, max_observer_polls_per_serial=128,
                observer_owners=4, channels=16, async_weight_max_tasks=4,
                retained_weight_H2D_passes=1, retention_D2H_passes=2,
                blocking_nonweight_copies=True, arithmetic_oracle_calls_during_runtime=0,
                semantic_snapshots=2 if baseline else 1,
                runtime_admitted=False)
