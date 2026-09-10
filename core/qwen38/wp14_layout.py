"""Fixed5PE WP14 buffer inventory and physical runtime operation sequence."""
from .qk_rope_protocol import ARRAYS as QK_ARRAYS, INITIAL, PROBES

PRODUCER_ARRAYS = {
    'weights_storage': (14338, 16), 'input_storage': (114, 32),
    'output_storage': (130, 32), 'bf16_storage': (130, 16),
    'control': (5, 32), 'state': (13, 32), 'timing': (276, 16),
    'weight_guard_snapshot': (5, 32),
}
HANDOFF_ARRAYS = {'handoff_control': (8, 32), 'handoff_state': (24, 32),
                  'wire_data': (140, 32), 'wire_ack': (10, 32)}
ARRAYS = {**PRODUCER_ARRAYS, **HANDOFF_ARRAYS,
          **{'qk_'+name: value for name, value in QK_ARRAYS.items()}}
QK_INITIAL = ('qk_input', *('qk_'+n for n in INITIAL), 'qk_control')
QK_READS = tuple('qk_'+n for n in QK_ARRAYS)


def sequence():
    rows = []
    def cp(direction, name, pe):
        count, bits = ARRAYS[name]
        rows.append(dict(kind='copy', direction=direction, name=name, pe=pe,
                         count=count, bits=bits, host_bytes=count*4, native_bytes=count*bits//8))
    def launch(name): rows.append(dict(kind='launch', name=name))
    for n in QK_INITIAL: cp('h2d', n, 4)
    launch('initialize_consumer')
    for n in ('qk_state', 'handoff_state'): cp('d2h', n, 4)
    for pe in range(4):
        for n in ('output_storage', 'bf16_storage'): cp('h2d', n, pe)
    for pe in range(4): cp('h2d', 'control', pe)
    launch('begin')
    for tile in range(46):
        for pe in range(4):
            for n in ('weights_storage', 'input_storage', 'control'): cp('h2d', n, pe)
        launch('accumulate')
        if tile in (0, 44):
            for pe in range(4):
                for n in ('output_storage', 'state'): cp('d2h', n, pe)
    launch('finalize')
    for pe in range(4):
        for n in ('output_storage', 'bf16_storage', 'state', 'weight_guard_snapshot',
                  'input_storage', 'timing'): cp('d2h', n, pe)
    for rank in range(4):
        for pe in range(5): cp('h2d', 'handoff_control', pe)
        launch('arm_handoff')
        cp('d2h', 'handoff_state', 4)
        launch('handoff')
        for pe in (rank, 4):
            for n in ('handoff_state', 'wire_data', 'wire_ack'): cp('d2h', n, pe)
        cp('d2h', 'qk_input', 4)
    launch('preprocess')
    for n in (*QK_READS, 'handoff_state'): cp('d2h', n, 4)
    # Verify retained producer outputs and full resident tiles after all fabric
    # and consumer work, before release. The weight read is moved, not repeated.
    for pe in range(4):
        for n in ('output_storage', 'bf16_storage', 'weights_storage'): cp('d2h', n, pe)
    launch('release')
    for pe in range(4): cp('d2h', 'state', pe)
    copies = launches = 0
    for r in rows:
        if r['kind'] == 'copy':
            copies += 1; r['sequence'] = copies
        else:
            launches += 1; r['ordinal'] = launches
    return rows


def budget():
    rows = sequence(); copies = [r for r in rows if r['kind'] == 'copy']
    return dict(copies=len(copies), launches=sum(r['kind'] == 'launch' for r in rows),
                host_bytes=sum(r['host_bytes'] for r in copies),
                payload_bytes=sum(r['native_bytes'] for r in copies),
                host_slots=sum(r['count'] for r in copies),
                max_host_buffer_bytes=max(r['host_bytes'] for r in copies),
                data_frames=4, data_frame_words=138, ack_frames=4, ack_frame_words=8,
                fabric_bytes=4*(138+8)*4)


def declared_bytes(role):
    if role not in ('producer', 'consumer'): raise ValueError('Unknown PE role')
    own = PRODUCER_ARRAYS if role == 'producer' else {'qk_'+n: v for n, v in QK_ARRAYS.items()}
    opposite = {'qk_'+n: v for n, v in QK_ARRAYS.items()} if role == 'producer' else PRODUCER_ARRAYS
    arrays = {**own, **HANDOFF_ARRAYS, **{n: (1, bits) for n, (_, bits) in opposite.items()}}
    # GEMV has512 internal bytes and two3-word timestamp arrays; consumer math
    # has no module-global arrays. Compiler temporaries/code/padding are separate.
    extra = 524 if role == 'producer' else 0
    return dict(arrays=arrays, explicit_kernel_and_timestamp_bytes=extra,
                total=sum(n*bits//8 for n, bits in arrays.values())+extra,
                excludes='Code, padding, compiler state, memcpy allocations and stack; actual ELF gate required.')
