"""Source-only four-PE partial-MLP interface and exact proposed host transaction ledger."""
from collections import Counter

SCOPE = 'mlp128-fullinput-silu-down128'
ROLES = ('gate', 'up', 'nonlinear', 'down_partial')
# Counts are device scalar elements per PE, including the declared guard words.
ARRAYS = {
    'weights_storage': (16, (14338, 14338, 1, 8194)),
    'input_storage': (32, (114, 114, 1, 66)),
    'output_storage': (32, (130, 130, 1, 130)),
    'bf16_storage': (16, (130, 130, 1, 130)),
    'control': (32, (8, 8, 8, 8)),
    'state': (32, (16, 16, 16, 16)),
    'timing': (16, (276, 276, 6, 12)),
    'weight_guard_snapshot': (32, (5, 5, 1, 5)),
    'tile_guard_snapshot': (32, (12, 12, 1, 12)),
    'handoff_control': (32, (10, 10, 10, 10)),
    'handoff_latched': (32, (10, 10, 10, 10)),
    'handoff_state': (32, (32, 32, 32, 32)),
    'wire_data': (32, (142, 142, 142, 142)),
    'wire_ack': (32, (15, 15, 15, 15)),
    'mlp_gate': (16, (1, 1, 130, 1)),
    'mlp_up': (16, (1, 1, 130, 1)),
    'mlp_silu': (16, (1, 1, 130, 1)),
    'mlp_product': (16, (1, 1, 130, 130)),
    'mlp_exp': (32, (1, 1, 130, 1)),
    'mlp_sigmoid': (32, (1, 1, 130, 1)),
    'mlp_silu_fp32': (32, (1, 1, 130, 1)),
    'mlp_product_fp32': (32, (1, 1, 130, 1)),
    'mlp_state': (32, (16, 16, 16, 16)),
    'release_control': (32, (4, 4, 4, 4)),
}
FLOAT32 = {'input_storage', 'output_storage', 'mlp_exp', 'mlp_sigmoid',
           'mlp_silu_fp32', 'mlp_product_fp32'}
EDGES = (
    dict(edge=0, sender=0, receiver=2, tensor='mlp_gate', role=0, data_color=2, ack_color=6,
         sender_queue=2, receiver_queue=2, forward_route=[0, 1, 2]),
    dict(edge=1, sender=1, receiver=2, tensor='mlp_up', role=1, data_color=3, ack_color=7,
         sender_queue=2, receiver_queue=3, forward_route=[1, 2]),
    dict(edge=2, sender=2, receiver=3, tensor='mlp_product', role=2, data_color=4, ack_color=8,
         sender_queue=4, receiver_queue=2, forward_route=[2, 3]),
)


def sequence():
    """No SDK import/launch. Host initialization writes metadata only to consumers."""
    result = []
    launches = 0
    def copy(direction, name, pe, phase, tile=None):
        bits, counts = ARRAYS[name]
        count = counts[pe]
        if count <= 1 or count * 4 > 65536:
            raise ValueError('Non-dummy bounded single-PE copy required')
        # Consumer numerical operands are initialized and filled on device only.
        if direction == 'h2d' and (name.startswith('mlp_') or (name == 'input_storage' and pe >= 2)):
            raise ValueError('Host intermediate injection is forbidden')
        result.append(dict(kind='copy', direction=direction, name=name, pe=pe, count=count,
            bits=bits, host_dtype='float32' if name in FLOAT32 else 'uint32',
            physical_host_bytes=count * 4, native_device_bytes=count * bits // 8,
            phase=phase, tile=tile,
            learned_parameter_payload=name == 'weights_storage'))
    def launch(name, phase):
        nonlocal launches
        launches += 1
        result.append(dict(kind='launch', name=name, ordinal=launches, phase=phase))
    def read(names, pe, phase, tile=None):
        for name in names:
            copy('d2h', name, pe, phase, tile)
    for pe in range(4):
        copy('h2d', 'control', pe, 'initialize')
    launch('initialize', 'initialize')
    for pe in range(4):
        read(('state', 'handoff_state'), pe, 'initialize')
    for pe in (0, 1):
        copy('h2d', 'control', pe, 'begin_projections')
    launch('begin_projections', 'begin_projections')
    for tile in range(46):
        for pe in (0, 1):
            for name in ('weights_storage', 'input_storage', 'control'):
                copy('h2d', name, pe, 'projection_tile', tile)
        launch('accumulate_projections', 'projection_tile')
        for pe in (0, 1):
            if tile in (0, 45):
                read(('weights_storage',), pe, 'projection_tile_readback', tile)
            read(('input_storage', 'output_storage', 'state', 'tile_guard_snapshot'), pe, 'projection_tile_readback', tile)
    launch('finalize_projections', 'finalize_projections')
    for pe in (0, 1):
        read(('output_storage', 'bf16_storage', 'state', 'timing', 'weight_guard_snapshot'), pe, 'finalize_projections')
    def handoff(edge):
        phase = 'handoff_' + str(edge['edge'])
        for pe in range(4):
            copy('h2d', 'handoff_control', pe, phase)
        launch('arm_handoff', phase)
        launch('handoff', phase)
        for pe in (edge['sender'], edge['receiver']):
            read(('wire_data', 'wire_ack', 'handoff_state', 'handoff_latched'), pe, phase)
        read((edge['tensor'], 'mlp_state'), edge['receiver'], phase)
    handoff(EDGES[0])
    handoff(EDGES[1])
    launch('evaluate_mlp', 'whole_silu_and_product')
    nonlinear = ('mlp_gate', 'mlp_up', 'mlp_silu', 'mlp_product', 'mlp_exp', 'mlp_sigmoid',
                 'mlp_silu_fp32', 'mlp_product_fp32', 'mlp_state')
    read((*nonlinear, 'timing'), 2, 'whole_silu_and_product')
    handoff(EDGES[2])
    copy('h2d', 'control', 3, 'begin_down')
    launch('begin_down', 'begin_down')
    for tile in range(2):
        for name in ('weights_storage', 'control'):
            copy('h2d', name, 3, 'down_tile', tile)
        launch('accumulate_down', 'down_tile')
        read(('weights_storage', 'input_storage', 'output_storage', 'state', 'tile_guard_snapshot'), 3, 'down_tile_readback', tile)
    launch('finalize_down', 'finalize_down')
    read(('output_storage', 'bf16_storage', 'state', 'timing', 'weight_guard_snapshot'), 3, 'finalize_down')
    for pe in (0, 1):
        read(('weights_storage', 'input_storage', 'output_storage', 'bf16_storage', 'state', 'handoff_state'), pe, 'retention_after_down')
    read((*nonlinear, 'handoff_state'), 2, 'retention_after_down')
    read(('mlp_product', 'weights_storage', 'input_storage', 'output_storage', 'bf16_storage', 'state', 'handoff_state'), 3, 'retention_after_down')
    for pe in range(4):
        copy('h2d', 'release_control', pe, 'release_after_retention')
    launch('release_all', 'release_after_retention')
    for pe in range(4):
        read(('state', 'handoff_state'), pe, 'released')
    for index, operation in enumerate(result):
        operation['sequence'] = index
    return result


def budget():
    operations = sequence()
    copies = [item for item in operations if item['kind'] == 'copy']
    weights = [item for item in copies if item['learned_parameter_payload']]
    return dict(scope=SCOPE, status='source_proposal_not_compiled_or_executed',
        physical_operations=len(operations), copy_calls=len(copies),
        launch_calls=sum(item['kind'] == 'launch' for item in operations),
        host_bytes=sum(item['physical_host_bytes'] for item in copies),
        native_bytes=sum(item['native_device_bytes'] for item in copies),
        maximum_host_buffer_bytes=max(item['physical_host_bytes'] for item in copies),
        copies_by_direction=dict(Counter(item['direction'] for item in copies)),
        weight_copies_by_direction=dict(Counter(item['direction'] for item in weights)),
        weight_host_bytes_by_direction={direction: sum(item['physical_host_bytes'] for item in weights if item['direction'] == direction) for direction in ('h2d', 'd2h')},
        source_weight_payload_bytes=2654208, new_network_bytes=0,
        data_fabric_words_per_edge=140, ack_fabric_words_per_edge=13,
        fabric_edge_words=3 * (140 + 13),
        fabric_word_hops=sum((len(edge['forward_route']) - 1) * (140 + 13) for edge in EDGES),
        declared_buffer_bytes_per_pe=[sum(bits * counts[pe] // 8 for bits, counts in ARRAYS.values()) for pe in range(4)],
        planned_imported_GEMV_expanded_scratch_bytes_per_pe=[512, 512, 512, 512],
        planned_private_timestamp_bytes_per_pe=[12, 12, 12, 12],
        planned_declared_and_kernel_buffer_bytes_per_pe=[
            sum(bits * counts[pe] // 8 for bits, counts in ARRAYS.values()) + 512 + 12
            for pe in range(4)],
        buffer_only_not_ELF_SRAM=True, actual_ELF_plus_stack_limit=49152,
        reserved_stack_bytes=4096, actual_ELF_required_before_simulation=True,
        observations_not_isolated_transfer_timing=True)


def tile_snapshot(pe, tile):
    """Exact pre/post-guard identity after one successful, uncommitted tile."""
    if pe not in (0, 1, 3) or type(tile) is not int or not 0 <= tile < (46 if pe < 2 else 2):
        raise ValueError('Known numeric PE and tile required')
    width, total = (112, 5120) if pe < 2 else (64, 128)
    before = width * tile
    valid = min(width, total - before)
    return [0xa55a, 0x5aa5, 0xa55a, 0x5aa5, 1, tile, before, valid,
            tile + 1, before + valid, 1, 1]
