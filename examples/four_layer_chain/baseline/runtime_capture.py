"""Four original layers and a fresh-runtime restored position, under one owner.

All arithmetic runs on the device. Injected IO must be bound to the complete
actual artifact and supervised by a separate deadline/cancellation owner.
"""
from pathlib import Path
import hashlib, time
from checkpoint import commit, identity
from runtime_boundary import Boundary, require
from runtime_timing import atomic_json


def capture(io, prepared, root, host_plan, lowered, families, profile, timing,
            artifact_sha256, prepared_pins, restore=None):
    import numpy as np
    root = Path(root)
    mode = 'baseline' if restore is None else 'restore'
    copies = host_plan['copies']
    budget = host_plan['baseline_budget' if restore is None else 'restored_budget']
    require(io.budget == budget, 'Actual finite schedule budget')
    gate = Boundary(lowered, families)
    schedule = [(1, 1, 0, 0), (2, 1, 1, 1), (3, 2, 0, 0)] if restore is None else [(1, 1, 1, 1)]
    completed, observations, edges, snapshots, metadata = [], [], [], [], []
    previous = {r['layer']: 0 for r in lowered['regions']}
    reset_signatures, previous_cache = {}, {}
    checkpoint_pin = None
    started = time.monotonic()

    def progress(phase, serial=0):
        atomic_json(root/'capture-progress.json', dict(mode=mode, phase=phase, serial=serial,
                    completed=completed, copies=io.copies, launches=io.launches,
                    host_bytes=io.host_bytes, raw_files=len(io.store.files), raw_bytes=io.store.raw_bytes))
        io.store.event('phase', phase=phase, serial=serial, mode=mode)

    def parameters(direction, label):
        weights_done = False
        for index, item in enumerate(copies['uploads']):
            if direction == 'h2d' and item['symbol'] == 'weights':
                if not weights_done:
                    from runtime_weights import Session
                    require(getattr(io, 'weight_session', None) is None, 'One strongly owned weight session')
                    io.weight_session = Session(io, prepared, root,
                        [p for p in copies['uploads'] if p['symbol'] == 'weights'], profile['weight_upload'])
                    io.weight_result = io.weight_session.run()
                    weights_done = True
                continue
            value, receipt = prepared.transfer(item)
            io.copy(direction, item, f'{label}-{index:04}', value=value,
                    prepared_receipt=receipt, retain_raw=False)
        progress(label)

    def states(serial, generation, position, phase):
        for index, item in enumerate(copies['state_copies']):
            value, _ = io.copy('d2h', item, f's{serial}-{phase}-state-{index:04}')
            gate.state(item, value, serial, generation, position, phase)
        io.store.event('all_layer_state_passed', serial=serial, generation=generation,
                       position=position, phase=phase)

    def persistent_boundary(label):
        for index, item in enumerate(copies['persistent_copies']):
            value, receipt = io.copy('d2h', item, f'{label}-persistent-{index:04}')
            if item['family'] == 'attention' and label == 'reset':
                require(receipt['sha256'] == previous_cache[identity(item)],
                        'Reset preserves the complete existing KV array')
            else:
                native = value.view('<u2') if item['bits'] == 16 else value.view('<u4')
                require(not np.any(native), 'Exact initially empty or reset linear state')

    def transport_fence(serial, resets):
        first = {}
        for snapshot in range(2):
            points = 0
            for index, item in enumerate(copies['transport_copies']):
                value, receipt = io.copy('d2h', item, f's{serial}-transport-{snapshot}-{index:04}')
                points += gate.transport(item, value, serial, resets)
                if snapshot == 0:
                    first[index] = receipt['sha256']
                else:
                    require(first[index] == receipt['sha256'], 'Two complete stable transport snapshots')
            require(points == 138040, 'Every application PE is in the transport fence')
        io.store.event('transport_fence_passed', serial=serial, coordinates=138040, snapshots=2)

    def semantic_snapshot(serial, generation, position):
        result = []
        for index, item in enumerate(copies['checkpoint_copies']):
            value, receipt = io.copy('d2h', item, f's{serial}-semantic-{index:03}')
            if item['symbol'] == 'checkpoint_control':
                gate.control(value, generation, position+1)
            result.append((item, receipt))
        snapshots.append(dict(serial=serial, generation=generation, position=position,
                              records=[dict(copy=item, receipt=receipt) for item, receipt in result]))
        return result

    try:
        parameters('h2d', 'upload-once')
        if restore is not None:
            for index, expected in enumerate(copies['checkpoint_copies']):
                item, value = restore.transfer(index)
                require(identity(item) == identity(expected), 'Complete ordered durable restore')
                io.copy('h2d', item, f'restore-bank-{index:03}', value=value)
            restore.verify_unchanged()
            io.store.event('all_restore_uploads_returned', copies=39, native_bytes=14378752,
                           blocking=True, initialize_called=False)
        io.launch('initialize')
        parameters('d2h', 'initial-retention')
        states(0, 1, 0 if restore is None else 1, 'initialized')
        if restore is None:
            persistent_boundary('initialized')
        for serial, generation, position, input_row in schedule:
            if generation == 2:
                require(completed == [1, 2], 'Reset only after both complete chain fences')
                io.launch('reset')
                states(2, 2, 0, 'reset')
                persistent_boundary('reset')
            require(copies['hidden_input']['layer_id'] == 0, 'Only original embedding input enters from host')
            io.copy('h2d', copies['hidden_input'], f's{serial}-original-input', value=prepared.input_row(input_row))
            io.launch('prepare')
            states(serial, generation, position, 'prepared')
            progress('prepared', serial)
            timing.begin_compute(serial)
            io.launch('compute')
            for item in copies['observers']:
                layer = item['layer_id']
                done = None
                for attempt in range(budget['max_observer_polls_per_serial']):
                    value, receipt = io.copy('d2h', item, f's{serial}-layer{layer}-observer-{attempt:03}')
                    observation = gate.observer(item, value, serial, previous[layer])
                    previous[layer] = observation['sequence']
                    if observation['complete']:
                        done = dict(layer=layer, serial=serial, attempts=attempt+1,
                                    receipt=receipt, observation=observation)
                        break
                    time.sleep(profile['observer_poll_interval_seconds'])
                require(done is not None, 'Finite complete observer budget for every original layer')
                observations.append(done)
            timing.observed()
            states(serial, generation, position, 'complete')
            transport_fence(serial, generation-1)
            for index, item in enumerate(copies['control_copies']):
                value, _ = io.copy('d2h', item, f's{serial}-control-{index:03}')
                gate.control(value, generation, position+1)
            for index, item in enumerate(copies['handoff_copies']):
                value, _ = io.copy('d2h', item, f's{serial}-handoff-{index:03}')
                gate.handoff(item, value, serial, generation, position)
            retained = {}
            for index, item in enumerate(copies['diagnostic_copies']):
                value, receipt = io.copy('d2h', item, f's{serial}-diagnostic-{index:04}')
                gate.lifecycle(item, value, serial)
                if item['dtype'] == 'f32':
                    require(np.all(np.isfinite(value)), 'Finite complete retained FP32 arithmetic')
                elif item['dtype'] == 'u16' and item['symbol'] != 'linear_receive_offsets':
                    require(not np.any((value & 0x7f80) == 0x7f80), 'Finite retained BF16 arithmetic')
                if item['symbol'] in ('hidden', 'norm_saved', 'observer', 'qk_archive', 'qk_archive_status'):
                    for y in range(item['height']):
                        for x in range(item['width']):
                            retained[item['symbol'], (item['x']+x, item['y']+y)] = value[y, x].copy()
                if item['symbol'] == 'cache':
                    previous_cache[identity(item)] = receipt['sha256']
                name = item['symbol']
                arithmetic = item['family'] == 'linear' and not (
                    name.endswith('counters') or name.startswith('linear_') or name == 'frame_counters')
                replay = arithmetic or name == 'hidden'
                if restore is None and replay:
                    if serial == 1:
                        reset_signatures[index] = receipt['sha256']
                    elif serial == 3:
                        require(reset_signatures[index] == receipt['sha256'], 'Exact original arithmetic reset replay')
            edge_report = gate.hidden_edges(retained)
            edges.append(dict(serial=serial, edges=edge_report))
            metadata.append(gate.attention_metadata(retained, serial, generation, position))
            if generation == 1:
                snapshot = semantic_snapshot(serial, generation, position)
                if restore is None and position == 0:
                    fence = dict(PEs=138040, serial=1, generation=1, next_position=1,
                                 transport_snapshots=2, stable=True, all_commands_complete=True,
                                 all_boundary_transfers_complete=True, hidden_edges=edge_report)
                    checkpoint_pin = commit(root, host_plan, lowered, snapshot, fence, artifact_sha256, prepared_pins)
            completed.append(serial)
            progress('serial-complete', serial)
        parameters('d2h', 'terminal-retention')
        preparation = prepared.verify_unchanged()
        actual_polls = sum(x['attempts'] for x in observations)
        expected = budget['max_copies'] - len(schedule)*4*128 + actual_polls
        require(io.copies == expected and io.launches == budget['max_launches'], 'Exact complete finite copy/launch schedule')
        result = dict(status='captured_pending_independent_numeric_and_restore_comparison', mode=mode,
                      completed=completed, observations=observations, hidden_edges=edges,
                      attention_metadata=metadata, semantic_snapshots=snapshots, checkpoint=checkpoint_pin,
                      artifact_sha256=artifact_sha256, prepared_pins=prepared_pins, preparation=preparation,
                      raw_files=io.store.files, raw_bytes=io.store.raw_bytes, copies=io.copies,
                      host_bytes=io.host_bytes, launches=io.launches, weight_upload=io.weight_result,
                      capture_seconds=time.monotonic()-started, full_model=False,
                      mathematical_audit_passed=False, actual_restore_qualified=False, CPU_forward=False)
        atomic_json(root/'capture.json', result)
        return result
    except BaseException as exc:
        atomic_json(root/'capture-failure.json', dict(error=type(exc).__name__, message=str(exc),
                    completed=completed, copies=io.copies, launches=io.launches,
                    raw_files=io.store.files, raw_bytes=io.store.raw_bytes))
        raise
