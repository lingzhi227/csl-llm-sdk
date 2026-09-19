"""One finite SDK run, three quiescent epochs and final actual API negatives."""
import json
import os
from pathlib import Path
import time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
from check_capture import check_epoch, check_negative
from capture_store import save_capture, save_progress
from reuse_gate import verify_reuse

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
SEQUENCE = 0


def event(kind, **fields):
    global SEQUENCE
    if SEQUENCE >= 96:
        raise ValueError('Finite journal bound')
    raw = (json.dumps(dict(sequence=SEQUENCE, seconds=time.monotonic() - START, kind=kind, **fields)) + '\n').encode()
    with (ROOT / 'journal.jsonl').open('ab') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    SEQUENCE += 1


def main():
    if sorted(os.sched_getaffinity(0)) != [0]:
        raise ValueError('CPU0 required')
    verify_reuse(ROOT, auxiliary='absent')
    if any((ROOT / name).exists() for name in ('captures', 'journal.jsonl', 'result.json', 'progress.json')):
        raise ValueError('One runtime attempt with new capture directory required')
    platform = get_platform(None, SimfabConfig(suppress_trace=True, num_threads=1, dump_core=False), SdkTarget.WSE3)
    runner = SdkRuntime('out', platform)
    result = dict(epochs=[], captures={})
    capture_files = {}
    counts = dict(copies=0, launches=0, host_slot_bytes=0)
    error = None
    stopped = False
    try:
        symbols = {name: runner.get_id(name) for name in ('evidence', 'qk', 'input', 'source')}

        def copy(label, symbol, box, count, half=False):
            x, y, width, height = box
            value = np.zeros(width * height * count, np.uint32)
            event('copy_enter', label=label, symbol=symbol, box=box, count=count,
                  host_bytes=int(value.nbytes), half=half)
            runner.memcpy_d2h(value, symbols[symbol], x, y, width, height, count,
                data_type=MemcpyDataType.MEMCPY_16BIT if half else MemcpyDataType.MEMCPY_32BIT,
                streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            answer = value.reshape(width * height, count).tolist()
            capture_files[label] = save_capture(ROOT, label, answer)
            result['captures'][label] = answer
            counts['copies'] += 1
            counts['host_slot_bytes'] += int(value.nbytes)
            event('copy_exit', label=label)
            save_progress(ROOT, capture_files, result['epochs'], counts, SEQUENCE)
            return answer

        def launch(name):
            event('launch_enter', name=name)
            runner.launch(name, nonblock=False)
            counts['launches'] += 1
            event('launch_exit', name=name)

        event('load_enter')
        runner.load()
        event('load_exit')
        runner.run()
        event('run_exit')
        launch('initialize')
        for epoch in (1, 2, 3):
            launch('prepare')
            launch('compute')
            evidence = copy(f'epoch{epoch}_evidence', 'evidence', [0, 0, 9, 5], 128)
            qk = copy(f'epoch{epoch}_qk', 'qk', [0, 0, 6, 1], 516, True)
            inputs = copy(f'epoch{epoch}_input', 'input', [0, 0, 6, 1], 1028, True)
            # All four native roots form one contiguous2x2 rectangle in ID order.
            sources = copy(f'epoch{epoch}_native_sources', 'source', [6, 1, 2, 2], 132, True)
            result['epochs'].append(check_epoch(evidence, qk, inputs, sources, epoch))
            save_progress(ROOT, capture_files, result['epochs'], counts, SEQUENCE)
        launch('negative_checks')
        result['negative'] = check_negative(copy('negative_evidence', 'evidence', [0, 0, 9, 5], 128), result['captures']['epoch3_evidence'])
        if counts != dict(copies=13, launches=8, host_slot_bytes=209664):
            raise ValueError('Exact operation/copy contract')
    except BaseException as exc:
        error = exc
        event('failure', message=str(exc)[:512])
    finally:
        event('stop_enter')
        try:
            runner.stop()
            stopped = True
            event('stop_exit')
        except BaseException as exc:
            if error is None:
                error = exc
        result.update(status='passed' if error is None else 'failed', normal_stop=stopped,
                      seconds=time.monotonic() - START, journal_events=SEQUENCE, **counts,
                      message=None if error is None else str(error)[:512])
        result['capture_files'] = capture_files
        raw = (json.dumps(result, separators=(',', ':')) + '\n').encode()
        if len(raw) > 524288:
            raise ValueError('Bounded numeric capture receipt')
        with (ROOT / 'result.json').open('xb') as stream:
            stream.write(raw)
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
