"""Record one diagnostic snapshot without host feedback between neural phases."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from fifo_observer_decode import decode_observation
from hw00.store import atomic_json
from contract import WIDTH, HEIGHT, EPOCHS, parameters, operations, spec, summary


def require(test, message):
    if not test:
        raise ValueError(message)


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_gate(value, epoch, prepared=False):
    data = value.reshape(HEIGHT, WIDTH, 12); expected = np.zeros_like(data)
    expected[:, :, 0] = 2*epoch-int(prepared); expected[:, :, 1] = epoch
    completed = epoch-int(prepared)
    expected[0, 0, 2] = 4*completed; expected[0, 16:40, 3] = completed
    require(np.array_equal(data, expected), 'Exact all-PE epoch/barrier/cache-position state')


def parameter_value(prepared, filename, box):
    value = np.load(prepared/filename, allow_pickle=False)
    if box is not None:
        x, y, w, h = box; value = value[y:y+h, x:x+w].copy()
    return value


def capture(runner, types, order, root):
    root = Path(root); plan = json.loads((root/'layer3-plan.json').read_bytes())
    inputs = json.loads((root/'inputs.json').read_bytes()); prepared = Path(inputs['prepared_root'])
    expected = list(operations(plan)); index = 0; sequence = 0
    evidence = root/'evidence'; evidence.mkdir(); started = time.monotonic()
    copies = launches = total_bytes = 0; files = {}; epochs = []; pending = {}; observation = {}
    stopped = False; primary = None
    names = {x['name'] for x in expected if x['kind'] == 'copy'}
    symbols = {}

    def event(kind, **details):
        nonlocal sequence
        require(sequence < 1024, 'Journal entry bound')
        require(not {'sequence', 'kind', 'seconds'} & details.keys(), 'Reserved journal fields')
        line = json.dumps(dict(sequence=sequence, kind=kind,
            seconds=time.monotonic()-started, **details))+'\n'
        with (evidence/'journal.jsonl').open('ab') as stream:
            stream.write(line.encode()); stream.flush(); os.fsync(stream.fileno())
        sequence += 1

    def operation(value):
        nonlocal index
        require(index < len(expected) and value == expected[index], 'Frozen operation sequence')
        index += 1

    def copy(direction, item, value=None):
        nonlocal copies
        operation(dict(kind='copy', direction=direction, **item))
        w, h, n = [item[k] for k in ('width', 'height', 'count')]
        dtype = np.float32 if item['dtype'] == 'f32' else np.uint32
        if value is None:
            value = np.zeros(w*h*n, dtype=dtype)
        value = np.ascontiguousarray(value, dtype=dtype).reshape(-1)
        require(value.size == w*h*n and value.nbytes <= 16 << 20, 'Bounded exact transfer shape')
        event('copy_enter', direction=direction, **item)
        opts = dict(data_type=types.MEMCPY_16BIT if item['bits'] == 16 else types.MEMCPY_32BIT,
            streaming=False, order=order.ROW_MAJOR, nonblock=False)
        if direction == 'h2d':
            runner.memcpy_h2d(symbols[item['name']], value, item['x'], item['y'], w, h, n, **opts)
        else:
            pending['inflight_'+item['name']] = value
            runner.memcpy_d2h(value, symbols[item['name']], item['x'], item['y'], w, h, n, **opts)
            del pending['inflight_'+item['name']]
        copies += 1; event('copy_exit')
        if item['bits'] == 16:
            value = (value & 65535).astype(np.uint16)
        return value.reshape(h, w, n)

    def launch(name):
        nonlocal launches
        operation(dict(kind='launch', name=name)); event('launch_enter', name=name)
        runner.launch(name, nonblock=False); launches += 1; event('launch_exit', name=name)

    def archive(name, arrays):
        nonlocal total_bytes
        path = evidence/name
        with path.open('xb') as stream:
            np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
        size = path.stat().st_size; total_bytes += size
        require(size <= 8 << 20 and total_bytes <= 16 << 20 and len(files) < 8, 'Saved evidence bounds')
        files[name] = dict(bytes=size, sha256=hash_file(path))

    def check_parameters():
        for i, (filename, item, box) in enumerate(parameters(plan)):
            original = parameter_value(prepared, filename, box).reshape(item['height'], item['width'], item['count'])
            actual = copy('d2h', item)
            require(actual.dtype == original.dtype and actual.shape == original.shape
                    and actual.tobytes() == original.tobytes(), 'Original resident parameter retention: '+str(i))

    try:
        symbols = {name: runner.get_id(name) for name in names}
        # Descriptor and route assertions occur inside initialize itself.
        for filename, item, box in parameters(plan):
            copy('h2d', item, parameter_value(prepared, filename, box))
        launch('initialize'); check_parameters()
        state = spec('state', 0, 0, WIDTH, HEIGHT, 12, 'u32')
        initial = copy('d2h', state); archive('initial.npz', dict(state=initial)); state_gate(initial, 0)
        rows = np.load(prepared/'reference-input.npy', allow_pickle=False)
        require(rows.shape == (8, 5120) and rows.dtype == np.uint16, 'Eight original boundary input rows')
        copy('h2d', spec('hidden', 1, 0, 1, 1, 5120, 'u16'), rows[0])
        launch('prepare')
        prepared_state = copy('d2h', state)
        archive('prepared.npz', dict(state=prepared_state))
        state_gate(prepared_state, 1, True)
        launch('compute')
        monitor_started = time.monotonic()
        atomic_json(root/'monitor-deadline.json', dict(active=True, deadline_monotonic=monitor_started+35))
        remaining = max(0, monitor_started+20-time.monotonic())
        event('observer_delay_enter', offset_seconds=20, seconds_requested=remaining)
        if remaining > 0:
            time.sleep(remaining)
        event('observer_delay_exit')
        require(time.monotonic() < monitor_started+35, 'Absolute observer deadline before read')
        raw = copy('d2h', spec('observer', 748, 0, 2, 25, 160, 'u32')).reshape(50, 160)
        require(time.monotonic() < monitor_started+35, 'Absolute observer deadline after read')
        archive('observer.npz', dict(raw=raw))
        observation = decode_observation(raw.tolist())
        observation['compute_seconds'] = time.monotonic()-monitor_started
        event('observer_snapshot_saved', **observation)
        atomic_json(root/'monitor-deadline.json', dict(active=False))
        require(index == len(expected), 'Complete finite operation sequence')
    except BaseException as error:
        primary = error
    finally:
        try:
            event('stop_enter');runner.stop(); stopped = True;event('stop_exit')
        except BaseException as error:
            if primary is None:
                primary = error
        if primary is not None and pending:
            try:
                archive('failed-partial.npz', pending)
            except BaseException as error:
                files['failure_retention_error'] = str(error)
        result = dict(status='captured' if primary is None else 'failed', normal_stop=stopped,
            complete_epochs=epochs, copies=copies, launches=launches, files=files,
            saved_bytes=total_bytes, seconds=time.monotonic()-started,
            error=None if primary is None else type(primary).__name__+': '+str(primary),
            original_layer=3, full_model=False, numerical_audit_passed=False,
            contract=summary(plan), observation=observation, diagnostic_only=True, between_phase_host_neural_feedback=False)
        with (root/'capture.json').open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if primary is not None:
        raise primary
    return result
