"""One finite simulator: idle backing, autonomous drain barrier, exact final data."""
import json
import os
import time
from pathlib import Path
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
from check_capture import initial, after_drain, final

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
SEQUENCE = 0


def event(kind, **fields):
    global SEQUENCE
    assert SEQUENCE < 40
    raw = (json.dumps(dict(sequence=SEQUENCE, seconds=time.monotonic()-START,
        kind=kind, **fields)) + '\n').encode()
    with (ROOT/'journal.jsonl').open('ab') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    SEQUENCE += 1


def main():
    assert sorted(os.sched_getaffinity(0)) == [0]
    platform = get_platform(None, SimfabConfig(suppress_trace=True, num_threads=1, dump_core=False), SdkTarget.WSE3)
    runner = SdkRuntime('out', platform)
    captures = {}; result = {}; error = None; stopped = False
    try:
        symbols = {name: runner.get_id(name) for name in
            ('evidence', 'fifo_backing', 'trace', 'sideband', 'packet_data')}
        options = dict(data_type=MemcpyDataType.MEMCPY_32BIT, streaming=False,
                       order=MemcpyOrder.ROW_MAJOR, nonblock=False)

        def copy(label, name, box, count):
            x, y, width, height = box
            value = np.zeros(width * height * count, np.uint32)
            event('copy_enter', label=label, symbol=name, box=box, count=count, words=int(value.size))
            runner.memcpy_d2h(value, symbols[name], x, y, width, height, count, **options)
            captures[label] = value.reshape(width * height, count).tolist()
            event('copy_exit', label=label)

        def launch(name):
            event('launch_enter', name=name)
            runner.launch(name, nonblock=False)
            event('launch_exit', name=name)

        event('load_enter'); runner.load(); event('load_exit')
        runner.run(); event('run_exit')
        launch('initialize')
        copy('initial', 'evidence', [0, 0, 8, 2], 160)
        copy('fifo0', 'fifo_backing', [0, 0, 5, 1], 64)
        copy('fifo1', 'fifo_backing', [0, 1, 5, 1], 128)
        initial(captures)
        launch('stall')
        copy('final_trace', 'trace', [2, 0, 1, 2], 160)
        copy('budget_trace', 'trace', [5, 0, 1, 2], 160)
        after_drain(captures)
        launch('coexist')
        copy('final', 'evidence', [0, 0, 8, 2], 160)
        copy('sideband', 'sideband', [6, 1, 1, 1], 32)
        copy('payload', 'packet_data', [0, 0, 8, 2], 31)
        result = final(captures)
    except BaseException as exc:
        error = exc; event('failure', message=str(exc)[:512])
    finally:
        event('stop_enter')
        try:
            runner.stop(); stopped = True; event('stop_exit')
        except BaseException as exc:
            if error is None:
                error = exc
        result.update(status='passed' if error is None else 'failed', normal_stop=stopped,
            captures=captures, seconds=time.monotonic()-START,
            message=None if error is None else str(error)[:512])
        with (ROOT/'result.json').open('x') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
