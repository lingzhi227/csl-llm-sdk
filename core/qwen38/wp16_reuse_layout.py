"""Connected four-PE exact source interface and transaction schedule."""
from collections import Counter

SCOPE = 'connected-fourPE-resident128x112-three-generations-partial-down128'
ARRAYS = {'weights_storage': (16, (14338, 14338, 1, 8194)),
 'input_storage': (32, (114, 114, 1, 66)),
 'output_storage': (32, (130, 130, 1, 130)),
 'bf16_storage': (16, (130, 130, 1, 130)),
 'control': (32, (8, 8, 8, 8)),
 'state': (32, (20, 20, 20, 20)),
 'timing': (16, (6, 6, 6, 12)),
 'weight_guard_snapshot': (32, (16, 16, 1, 16)),
 'tile_guard_snapshot': (32, (16, 16, 1, 16)),
 'handoff_control': (32, (12, 12, 12, 12)),
 'handoff_latched': (32, (12, 12, 12, 12)),
 'handoff_state': (32, (34, 34, 34, 34)),
 'wire_data': (32, (144, 144, 144, 144)),
 'wire_ack': (32, (17, 17, 17, 17)),
 'mlp_gate': (16, (1, 1, 130, 1)),
 'mlp_up': (16, (1, 1, 130, 1)),
 'mlp_silu': (16, (1, 1, 130, 1)),
 'mlp_product': (16, (1, 1, 130, 130)),
 'mlp_exp': (32, (1, 1, 130, 1)),
 'mlp_sigmoid': (32, (1, 1, 130, 1)),
 'mlp_silu_fp32': (32, (1, 1, 130, 1)),
 'mlp_product_fp32': (32, (1, 1, 130, 1)),
 'mlp_state': (32, (20, 20, 20, 20)),
 'release_control': (32, (12, 12, 12, 12)),
 'generation_control': (32, (12, 12, 12, 12)),
 'lifecycle_state': (32, (32, 32, 32, 32))}
FLOAT32 = {'mlp_product_fp32', 'input_storage', 'output_storage', 'mlp_exp', 'mlp_silu_fp32', 'mlp_sigmoid'}
EDGES = [{'edge': 0,
  'sender': 0,
  'receiver': 2,
  'tensor': 'mlp_gate',
  'role': 0,
  'data_color': 2,
  'ack_color': 6,
  'sender_queue': 2,
  'receiver_queue': 2,
  'forward_route': [0, 1, 2]},
 {'edge': 1,
  'sender': 1,
  'receiver': 2,
  'tensor': 'mlp_up',
  'role': 1,
  'data_color': 3,
  'ack_color': 7,
  'sender_queue': 2,
  'receiver_queue': 3,
  'forward_route': [1, 2]},
 {'edge': 2,
  'sender': 2,
  'receiver': 3,
  'tensor': 'mlp_product',
  'role': 2,
  'data_color': 4,
  'ack_color': 8,
  'sender_queue': 4,
  'receiver_queue': 2,
  'forward_route': [2, 3]}]
FIELD_INDICES = {'handoff_control': {'generation': 0,
                     'input_index': 1,
                     'contraction_id': 2,
                     'edge_id': 3,
                     'tensor_role': 4,
                     'channel_start': 5,
                     'channel_count': 6,
                     'output_start': 7,
                     'output_count': 8,
                     'payload_words': 9,
                     'projection_input_start': 10,
                     'projection_input_count': 11},
 'handoff_latched': {'generation': 0,
                     'input_index': 1,
                     'contraction_id': 2,
                     'edge_id': 3,
                     'tensor_role': 4,
                     'channel_start': 5,
                     'channel_count': 6,
                     'output_start': 7,
                     'output_count': 8,
                     'payload_words': 9,
                     'projection_input_start': 10,
                     'projection_input_count': 11},
 'control': {'generation': 0,
             'input_index': 1,
             'contraction_id': 2,
             'total_columns': 3,
             'tile_index': 4,
             'column_offset': 5,
             'valid_columns': 6,
             'output_start': 7},
 'state': {'phase': 0,
           'generation': 1,
           'input_index': 2,
           'contraction_id': 3,
           'total_columns': 4,
           'tile_count': 5,
           'consumed_columns': 6,
           'commit_id': 7,
           'error': 8,
           'last_valid_columns': 9,
           'accumulate_count': 10,
           'begin_count': 11,
           'finalize_count': 12,
           'release_count': 13,
           'pe_role': 14,
           'output_start': 15,
           'projection_input_start': 16,
           'projection_input_count': 17,
           'output_count': 18,
           'reserved_zero': 19},
 'handoff_state': {'phase': 0,
                   'generation': 1,
                   'input_index': 2,
                   'contraction_id': 3,
                   'edge_id': 4,
                   'committed_payload_mask': 5,
                   'command_seen': 6,
                   'receipt_complete': 7,
                   'ack_sent': 8,
                   'ack_received': 9,
                   'data_send_complete': 10,
                   'command_unblocked': 11,
                   'error': 12,
                   'data_words': 13,
                   'ack_words': 14,
                   'commit_count': 15,
                   'event_counter': 16,
                   'arm_event': 17,
                   'command_event': 18,
                   'receipt_event': 19,
                   'commit_event': 20,
                   'ack_sent_event': 21,
                   'ack_received_event': 22,
                   'unblock_event': 23,
                   'tensor_role': 24,
                   'channel_start': 25,
                   'channel_count': 26,
                   'output_start': 27,
                   'output_count': 28,
                   'release_count': 29,
                   'receipt_before_command': 30,
                   'retained': 31,
                   'projection_input_start': 32,
                   'projection_input_count': 33},
 'mlp_state': {'phase': 0,
               'generation': 1,
               'input_index': 2,
               'contraction_id': 3,
               'incoming_mask': 4,
               'silu_count': 5,
               'product_count': 6,
               'down_input_count': 7,
               'down_tile_count': 8,
               'down_consumed': 9,
               'output_count': 10,
               'complete_epoch': 11,
               'error': 12,
               'release_count': 13,
               'pe_role': 14,
               'reserved_zero': 15,
               'projection_input_start': 16,
               'projection_input_count': 17,
               'output_start': 18,
               'scope_output_count': 19},
 'tile_guard_snapshot': {'head_before': 0,
                         'tail_before': 1,
                         'head_after': 2,
                         'tail_after': 3,
                         'generation': 4,
                         'tile_index': 5,
                         'consumed_before': 6,
                         'valid_columns': 7,
                         'completed_tiles': 8,
                         'consumed_after': 9,
                         'pre_guard_ok': 10,
                         'post_guard_ok': 11,
                         'input_index': 12,
                         'contraction_id': 13,
                         'total_columns': 14,
                         'output_start': 15},
 'generation_control': {'generation': 0,
                        'input_index': 1,
                        'contraction_id': 2,
                        'projection_input_start': 3,
                        'projection_input_count': 4,
                        'channel_start': 5,
                        'channel_count': 6,
                        'output_start': 7,
                        'output_count': 8,
                        'expected_previous_released_generation': 9,
                        'begin_opcode': 10,
                        'reserved_zero': 11},
 'release_control': {'generation': 0,
                     'input_index': 1,
                     'contraction_id': 2,
                     'projection_input_start': 3,
                     'projection_input_count': 4,
                     'channel_start': 5,
                     'channel_count': 6,
                     'output_start': 7,
                     'output_count': 8,
                     'required_local_completion_mask': 9,
                     'release_ordinal': 10,
                     'reserved_zero': 11},
 'weight_guard_snapshot': {'left_guard': 0,
                           'right_guard': 1,
                           'generation': 2,
                           'input_index': 3,
                           'contraction_id': 4,
                           'total_columns': 5,
                           'output_start': 6,
                           'output_count': 7,
                           'phase': 8,
                           'tile_count': 9,
                           'consumed_columns': 10,
                           'commit_id': 11,
                           'boundary_serial': 12,
                           'check_success': 13,
                           'pe_role': 14,
                           'reserved_zero': 15},
 'lifecycle_state': {'phase': 0,
                     'generation': 1,
                     'input_index': 2,
                     'contraction_id': 3,
                     'projection_input_start': 4,
                     'projection_input_count': 5,
                     'channel_start': 6,
                     'channel_count': 7,
                     'output_start': 8,
                     'output_count': 9,
                     'previous_released_generation': 10,
                     'expected_next_generation': 11,
                     'total_reset_count': 12,
                     'total_begin_count': 13,
                     'total_release_count': 14,
                     'next_permitted_edge': 15,
                     'reset_check_mask': 16,
                     'reset_check_count': 17,
                     'resident_guard_check_count': 18,
                     'error': 19,
                     'active_data_transfer': 20,
                     'active_ACK_transfer': 21,
                     'total_data_send_completions': 22,
                     'total_ACK_receipts': 23,
                     'total_data_receipts': 24,
                     'total_ACK_send_completions': 25,
                     'retained_operand_mask': 26,
                     'last_release_generation': 27,
                     'local_completion_mask': 28,
                     'reset_event_counter': 29,
                     'release_event_counter': 30,
                     'pe_role': 31}}


def sequence():
    arrays = {name: dict(bits=bits, counts=counts) for name,(bits,counts) in ARRAYS.items()}
    ops = []
    generation = 0
    input_index = None
    def copy(direction, name, pe, phase, tile=None):
        value = arrays[name]; count = value['counts'][pe]; bits = value['bits']
        assert count > 1 and count * 4 <= 65536
        assert not (direction == 'h2d' and (name.startswith('mlp_') or name == 'input_storage' and pe >= 2))
        ops.append(dict(kind='copy', direction=direction, name=name, pe=pe,
            count=count, bits=bits, host_dtype='float32' if name in FLOAT32 else 'uint32',
            physical_host_bytes=count*4, native_device_bytes=count*bits//8,
            generation=generation, input_index=input_index, phase=phase, tile=tile,
            learned_parameter_payload=name == 'weights_storage'))
    def readback(names, pe, phase, tile=None):
        for name in names: copy('d2h', name, pe, phase, tile)
    def launch(name, phase=None):
        ops.append(dict(kind='launch', name=name, phase=phase or name,
            generation=generation, input_index=input_index))
    nonlinear = ('mlp_gate', 'mlp_up', 'mlp_silu', 'mlp_product', 'mlp_exp',
                 'mlp_sigmoid', 'mlp_silu_fp32', 'mlp_product_fp32', 'mlp_state')
    for pe in range(4): copy('h2d', 'control', pe, 'initialize_epoch0')
    launch('initialize', 'initialize_epoch0')
    for pe in range(4):
        readback(('state', 'handoff_state', 'mlp_state', 'lifecycle_state'), pe, 'initialize_epoch0')
    for pe in (0, 1):
        copy('h2d', 'weights_storage', pe, 'resident_weights_load_once', 0)
        readback(('weights_storage',), pe, 'resident_weights_before_generation1', 0)

    for generation, input_index in enumerate((0, 1, 3), start=1):
        for pe in range(4): copy('h2d', 'generation_control', pe, 'begin_generation')
        launch('begin_generation')
        for pe in range(4):
            readback(('state', 'handoff_state', 'mlp_state', 'lifecycle_state',
                      'wire_data', 'wire_ack', 'handoff_latched'), pe, 'reset_observation')
        for pe in (0, 1, 3):
            readback(('input_storage', 'output_storage', 'bf16_storage',
                      'tile_guard_snapshot', 'weight_guard_snapshot', 'timing'), pe, 'reset_observation')
        readback((*nonlinear[:-1], 'timing'), 2, 'reset_observation')
        readback(('mlp_product',), 3, 'reset_observation')
        for pe in (0, 1): copy('h2d', 'control', pe, 'begin_projections')
        launch('begin_projections')
        for pe in (0, 1):
            for name in ('input_storage', 'control'): copy('h2d', name, pe, 'projection_tile', 0)
        launch('accumulate_projections', 'projection_tile')
        for pe in (0, 1):
            readback(('input_storage', 'output_storage', 'state', 'tile_guard_snapshot'), pe, 'projection_tile_readback', 0)
        launch('finalize_projections')
        for pe in (0, 1):
            readback(('output_storage', 'bf16_storage', 'state', 'timing', 'weight_guard_snapshot'), pe, 'finalize_projections')
        def handoff(edge, sender, receiver, tensor):
            phase = 'handoff_' + str(edge)
            for pe in range(4): copy('h2d', 'handoff_control', pe, phase)
            launch('arm_handoff', phase); launch('handoff', phase)
            for pe in (sender, receiver):
                readback(('wire_data', 'wire_ack', 'handoff_state', 'handoff_latched'), pe, phase)
            readback((tensor, 'mlp_state'), receiver, phase)
        handoff(0, 0, 2, 'mlp_gate'); handoff(1, 1, 2, 'mlp_up')
        launch('evaluate_mlp', 'whole_silu_and_product')
        readback((*nonlinear, 'timing'), 2, 'whole_silu_and_product')
        handoff(2, 2, 3, 'mlp_product')
        copy('h2d', 'control', 3, 'begin_down'); launch('begin_down')
        for tile in (0, 1):
            for name in ('weights_storage', 'control'): copy('h2d', name, 3, 'down_tile', tile)
            launch('accumulate_down', 'down_tile')
            readback(('weights_storage', 'input_storage', 'output_storage', 'state', 'tile_guard_snapshot'), 3, 'down_tile_readback', tile)
        launch('finalize_down')
        readback(('output_storage', 'bf16_storage', 'state', 'timing', 'weight_guard_snapshot'), 3, 'finalize_down')
        for pe in (0, 1):
            readback(('input_storage', 'output_storage', 'bf16_storage', 'state', 'handoff_state'), pe, 'retention_after_down')
        readback((*nonlinear, 'handoff_state'), 2, 'retention_after_down')
        readback(('mlp_product', 'weights_storage', 'input_storage', 'output_storage', 'bf16_storage', 'state', 'handoff_state'), 3, 'retention_after_down')
        for pe in range(4): readback(('lifecycle_state',), pe, 'retention_after_down')
        for pe in range(4): copy('h2d', 'release_control', pe, 'release_after_retention')
        launch('release_all', 'release_after_retention')
        for pe in range(4):
            readback(('state', 'handoff_state', 'mlp_state', 'lifecycle_state'), pe, 'released_join')
    for pe in (0, 1): readback(('weights_storage',), pe, 'resident_weights_after_generation3', 0)
    for index, op in enumerate(ops): op['sequence'] = index
    return ops


def budget():
    ops=sequence();copies=[o for o in ops if o['kind']=='copy']
    return dict(scope=SCOPE, status='source_implementation_not_compiled_or_executed',
        physical_operations=len(ops),copy_calls=len(copies),launch_calls=len(ops)-len(copies),
        copies_by_direction=dict(Counter(o['direction'] for o in copies)),
        host_bytes=sum(o['physical_host_bytes'] for o in copies),
        native_bytes=sum(o['native_device_bytes'] for o in copies),
        raw_D2H_bytes=sum(o['physical_host_bytes'] for o in copies if o['direction']=='d2h'),
        maximum_host_buffer_bytes=max(o['physical_host_bytes'] for o in copies),
        declared_buffer_bytes_per_PE=[sum(bits*counts[pe]//8 for bits,counts in ARRAYS.values()) for pe in range(4)],
        actual_ELF_plus_stack_limit=49152,reserved_stack_bytes=4096,actual_ELF_required_before_simulation=True,
        generation_input_contraction=[[1,0,1],[2,1,2],[3,3,3]],data_fabric_words_per_edge=142,
        ack_fabric_words_per_edge=15,fabric_words=1413,fabric_word_hops=1884,
        planned_simulator_seconds=400,hard_simulator_seconds=420,observations_include_queued_work=True)
