"""Strict synthetic archive/alias equivalence gates, not full layer acceptance."""
from copy import deepcopy
from archive_decode import decode_archive

PLACEMENTS = {(x, 0): x for x in range(4)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def words(value, count):
    require(len(value) == count and all(type(x) is int and 0 <= x <= 0xffffffff for x in value),
            'Exact unsigned capture width')


def check_epoch(evidence, archive, reference, workspace, output, baseline_output, epoch):
    require(epoch in (1, 2) and len(evidence) == 4, 'Two finite reset generations, four PEs')
    for role, row in enumerate(evidence):
        words(row, 64)
        require(row[0] == 0 and row[1:4] == [role, epoch, 10+epoch]
                and row[4] == 1 and row[6:8] == [0, 0] and row[48:51] == [0, 0, 0],
                'Quiescent actual PE identity and zero errors')
    for role in (0, 3):
        require(evidence[role][5] == 1 and evidence[role][16:21] == [0, 1, 2, 7, 7],
                'Original QK/attention returned without error and output was consumed')
    for role in (0, 1):
        require(evidence[role][8:15] == [8*epoch, 7*epoch, epoch, 5, 2, 0, 0],
                'Exact source/sink archive and metadata counts')
    require(evidence[0][41] == 1 and evidence[2][16:18] == [1, 1],
            'Source overwrite and both actual release notices')
    require(evidence[1][32:37] == [16*epoch, 11, 1 << 11, 1, 1]
            and evidence[1][37] in (123, 132), 'Actual pending/final/commit callback order')
    words(archive, 960)
    words(reference, 960)
    words(workspace, 1280)
    require(archive == reference, 'Every archived bit equals the independent unaliased original owner')
    require(workspace == [0xdead0000 | epoch] * 1280,
            'Entire source workspace was overwritten before sink commit release')
    decoded = decode_archive(evidence[1][16:32], archive, head=17,
                             identity=[7, 10+epoch, 0, 2, 0], generation=epoch)
    require(decoded['archive_marker_sequence'] == 8*(epoch-1)+5, 'Exact private archive sequence')
    for record in decoded['arrays'].values():
        mask = 0x7f800000 if record['dtype'] == 'f32' else 0x7f80
        require(all((v & mask) != mask for v in record['bits']), 'All synthetic QK diagnostics finite')
    for vector in (output, baseline_output):
        words(vector, 256)
        require(all(v <= 65535 and (v & 0x7f80) != 0x7f80 for v in vector),
                'Actual BF16 slots have zero high bits and finite values')
        require(any(v & 0x7fff for v in vector), 'Nonzero synthetic attention output')
    require(output == baseline_output, 'All 256 BF16 outputs equal the unaliased original owner')
    return dict(status='passed',epoch=epoch, request=7, reset=10+epoch, position=0, head=17,
                archive_u32_equal=960, output_BF16_equal=256, source_workspace_poisoned_u32=1280,
                actual_sink_callback_order=evidence[1][37], actual_sink_committed=True,
                scope='Synthetic alias/archive and original-kernel equivalence only')


def check_negative(evidence, before):
    expected = deepcopy(before)
    expected[2][48:51] = [60, 0, 170]
    expected[1][48:51] = [1, 0, 2]
    expected[1][13] = 32768
    expected[1][14] = 1
    expected[1][30] = 32768
    require(evidence == expected, 'Exact actual API rejection counters and late-error isolation')
    return dict(status='passed',actual_rejected_calls=61, actual_positive_calls=172,
                late_error_invalidates_existing_commit=True,
                final_before_commit_and_commit_before_final='Shared device predicate with isolated actual lifecycle API calls')
