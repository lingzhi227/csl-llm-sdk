"""Recheck all physical boundaries independently after owner release."""
from pathlib import Path
import hashlib
from checkpoint import identity, _payload
from runtime_boundary import Boundary, require
from runtime_schedule import schedule


def audit(evidence, host, lowered, families, prepared, restore=None):
    import numpy as np
    copies, captured = host['copies'], evidence.capture
    mode = captured['mode']
    require((restore is None) == (mode == 'baseline'), 'Exact restoration source is independently bound')
    gate = Boundary(lowered, families)
    previous = {layer: 0 for layer in range(4)}
    observations = {(o['serial'], o['layer']): o for o in captured['observations']}
    reset_hashes, cache_hashes, edge_reports = {}, {}, []

    def raw(name, item):
        return evidence.checked(name+'.bin', item)

    def equal(a, b, message):
        require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), message)

    def states(serial, generation, position, phase):
        for index, item in enumerate(copies['state_copies']):
            gate.state(item, raw(f's{serial}-{phase}-state-{index:04}', item),
                       serial, generation, position, phase)

    if restore is not None:
        for index, expected in enumerate(copies['checkpoint_copies']):
            item, value = restore.transfer(index)
            require(identity(item) == identity(expected), 'Exact complete restoration inventory')
            equal(raw(f'restore-bank-{index:03}', item), value, 'Every uploaded semantic byte is the durable checkpoint byte')
        restore.verify_unchanged()
    states(0, 1, 0 if mode == 'baseline' else 1, 'initialized')
    if mode == 'baseline':
        for index, item in enumerate(copies['persistent_copies']):
            value = raw(f'initialized-persistent-{index:04}', item)
            require(not np.any(value.view('<u2') if item['bits'] == 16 else value.view('<u4')),
                    'All original persistent arrays start exactly empty')
    semantic_reports = []
    for serial, generation, position, input_row in schedule(mode):
        if generation == 2:
            states(2, 2, 0, 'reset')
            for index, item in enumerate(copies['persistent_copies']):
                value = raw(f'reset-persistent-{index:04}', item)
                if item['family'] == 'attention':
                    require(hashlib.sha256(value.tobytes()).hexdigest() == cache_hashes[identity(item)],
                            'Reset keeps every existing KV byte')
                else:
                    require(not np.any(value.view('<u2') if item['bits'] == 16 else value.view('<u4')),
                            'Reset clears every linear persistent state bit')
        equal(raw(f's{serial}-original-input', copies['hidden_input']), prepared.input_row(input_row),
              'Only the pinned original Layer0 embedding enters normal execution')
        states(serial, generation, position, 'prepared')
        states(serial, generation, position, 'complete')
        for item in copies['observers']:
            layer = item['layer_id']
            observed = observations[serial, layer]
            for attempt in range(observed['attempts']):
                name = f's{serial}-layer{layer}-observer-{attempt:03}'
                result = gate.observer(item, raw(name, item), serial, previous[layer])
                previous[layer] = result['sequence']
                require(result['complete'] == (attempt == observed['attempts']-1),
                        'Every layer stops polling at its first coherent completion')
            require(result == observed['observation'] and
                    captured['raw_files'][name+'.bin'] == observed['receipt'], 'Final observation is raw-backed')
        points = 0
        for index, item in enumerate(copies['transport_copies']):
            first = raw(f's{serial}-transport-0-{index:04}', item)
            second = raw(f's{serial}-transport-1-{index:04}', item)
            points += gate.transport(item, first, serial, generation-1)
            gate.transport(item, second, serial, generation-1)
            equal(first, second, 'Two complete stable actual transport fences')
        require(points == 138040, 'All application PEs independently fenced')
        for index, item in enumerate(copies['control_copies']):
            gate.control(raw(f's{serial}-control-{index:03}', item), generation, position+1)
        for index, item in enumerate(copies['handoff_copies']):
            gate.handoff(item, raw(f's{serial}-handoff-{index:03}', item), serial, generation, position)
        retained, diagnostics = {}, {}
        for index, item in enumerate(copies['diagnostic_copies']):
            name = f's{serial}-diagnostic-{index:04}'
            value = raw(name, item)
            gate.lifecycle(item, value, serial)
            if item['dtype'] == 'f32':
                require(np.all(np.isfinite(value)), 'All captured FP32 arithmetic finite')
            elif item['dtype'] == 'u16' and item['symbol'] != 'linear_receive_offsets':
                require(not np.any((value & 0x7f80) == 0x7f80), 'All captured BF16 arithmetic finite')
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            diagnostics[identity(item)] = digest
            if item['symbol'] in ('hidden', 'norm_saved', 'observer', 'qk_archive', 'qk_archive_status'):
                for y in range(item['height']):
                    for x in range(item['width']):
                        retained[item['symbol'], (item['x']+x, item['y']+y)] = value[y, x].copy()
            if item['symbol'] == 'cache':
                cache_hashes[identity(item)] = digest
            name = item['symbol']
            replay = name == 'hidden' or item['family'] == 'linear' and not (
                name.endswith('counters') or name.startswith('linear_') or name == 'frame_counters')
            if mode == 'baseline' and replay:
                if serial == 1:
                    reset_hashes[index] = digest
                elif serial == 3:
                    require(reset_hashes[index] == digest, 'Every linear neural array and all four hidden outputs replay exactly')
        edges = gate.hidden_edges(retained)
        edge_reports.append(dict(serial=serial, edges=edges))
        metadata = gate.attention_metadata(retained, serial, generation, position)
        require(metadata in captured['attention_metadata'], 'Raw-backed complete QK/head metadata')
        if generation == 1:
            selected = [s for s in captured['semantic_snapshots'] if s['serial'] == serial]
            require(len(selected) == 1 and selected[0]['generation'] == generation and selected[0]['position'] == position,
                    'Exactly one semantic snapshot at the logical position')
            records = selected[0]['records']
            require(len(records) == 39, 'Complete semantic snapshot inventory')
            for index, (item, record) in enumerate(zip(copies['checkpoint_copies'], records)):
                require(identity(item) == identity(record['copy']) and record['receipt'] ==
                        captured['raw_files'][f's{serial}-semantic-{index:03}.bin'], 'Raw-backed ordered semantic snapshot')
                value = raw(f's{serial}-semantic-{index:03}', item)
                if item['symbol'] == 'checkpoint_control':
                    gate.control(value, generation, position+1)
                else:
                    require(hashlib.sha256(value.tobytes()).hexdigest() == diagnostics[identity(item)],
                            'Every semantic array equals the independently audited retained diagnostic')
            semantic_reports.append(dict(serial=serial, position=position, files=39, native_bytes=14378752))
    require(edge_reports == captured['hidden_edges'], 'All physical hidden edge claims independently reproduced')
    return dict(status='all_actual_control_and_boundary_checks_passed', mode=mode,
                all_states=True, every_PE_transport=True, full_hidden_edges=edge_reports,
                semantic_snapshots=semantic_reports, own_original_inputs=True,
                restore_H2D_bit_exact=restore is not None, full_model=False)
