"""One synthetic phase0 launch, one explicit core, and no post-compute memcpy."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)

ROOT = Path(__file__).resolve().parent
START = time.monotonic()
SEQUENCE = 0

def event(kind, **fields):
    global SEQUENCE
    assert SEQUENCE < 32
    raw = (json.dumps(dict(sequence=SEQUENCE, seconds=time.monotonic()-START,
                           kind=kind, **fields))+'\n').encode()
    with (ROOT/'journal.jsonl').open('ab') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    SEQUENCE += 1

def core_inventory():
    files = sorted(ROOT.glob('corefile.cs1*'))
    allowed = {'corefile.cs1.idx'} | {'corefile.cs1_'+str(i)+'.cgz' for i in range(8)}
    assert {p.name for p in files} == allowed and len(files) == 9
    total = 0
    result = {}
    for p in files:
        assert p.name in allowed and p.is_file() and not p.is_symlink()
        size = p.stat().st_size
        assert 0 < size <= 67108864
        total += size
        assert total <= 100663296
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1048576), b''):
                digest.update(block)
        result[p.name] = dict(bytes=size, sha256=digest.hexdigest())
    return result

def main():
    assert sorted(os.sched_getaffinity(0)) == [0]
    plan = json.loads((ROOT/'plan.json').read_bytes())
    assert plan['application'] == [37,10] and plan['fabric'] == [44,12]
    metadata = np.asarray(plan['metadata'], dtype=np.uint32)
    assert metadata.shape == (10,37,6) and np.max(metadata) < 65536
    expected_state = np.zeros((10,37,16), np.uint32)
    expected_state[:,:,0] = 1
    expected_state[:,:,1] = metadata[:,:,0]
    initial = np.zeros_like(expected_state)
    read_metadata = np.zeros_like(metadata)
    assert not list(ROOT.glob('corefile.cs1*'))
    assert json.loads((ROOT/'compiled.json').read_bytes())['passed']
    platform = get_platform(None, SimfabConfig(suppress_trace=True, num_threads=1,
                           dump_core=False), SdkTarget.WSE3)
    runner = SdkRuntime('out', platform)
    normal_stop = False
    error = None
    result = dict(post_compute_memcpy_calls=0, copies=0, launches=0, explicit_core_calls=0, core_files={})
    options = dict(streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    try:
        metadata_id = runner.get_id('metadata')
        state_id = runner.get_id('state')
        event('load_enter'); runner.load(); event('load_exit')
        runner.run(); event('run_exit')
        event('metadata_upload_enter')
        runner.memcpy_h2d(metadata_id, metadata.reshape(-1), 0,0,37,10,6,
                         data_type=MemcpyDataType.MEMCPY_16BIT, **options)
        result['copies'] += 1
        event('metadata_upload_exit')
        event('initialize_enter'); runner.launch('initialize', nonblock=False)
        result['launches'] += 1
        event('initialize_exit')
        event('metadata_read_enter')
        runner.memcpy_d2h(read_metadata.reshape(-1), metadata_id, 0,0,37,10,6,
                         data_type=MemcpyDataType.MEMCPY_16BIT, **options)
        result['copies'] += 1
        event('metadata_read_exit')
        assert np.array_equal(read_metadata & np.uint32(65535), metadata)
        # This all-PE barrier proves every collector is armed before roots start.
        event('initial_state_read_enter')
        runner.memcpy_d2h(initial.reshape(-1), state_id, 0,0,37,10,16,
                         data_type=MemcpyDataType.MEMCPY_32BIT, **options)
        result['copies'] += 1
        event('initial_state_read_exit')
        assert np.array_equal(initial, expected_state)
        event('compute_enter'); runner.launch('compute', nonblock=False)
        result['launches'] += 1
        event('compute_exit')
        # No post-compute memcpy, polling command, or SDK idle-time reset pulse.
        time.sleep(85)
        event('dump_core_enter'); result['explicit_core_calls'] += 1
        runner.dump_core('corefile.cs1')
        event('dump_core_exit')
    except BaseException as exc:
        error = exc
        event('failure', message=str(exc)[:512])
    finally:
        event('stop_enter')
        try:
            runner.stop(); normal_stop = True; event('stop_exit')
        except BaseException as exc:
            if error is None: error = exc
        # Archive only already captured host initialization values after stop.
        np.savez(ROOT/'initial.npz', metadata=read_metadata, state=initial)
        if error is None:
            try: result['core_files'] = core_inventory()
            except BaseException as exc: error = exc
        result.update(status='captured' if error is None else 'failed',
                      normal_stop=normal_stop, seconds=time.monotonic()-START,
                      hardware=False, neural_math=False, host_slot_bytes=41440,
                      completion_claimed=False,
                      message=None if error is None else str(error)[:512])
        with (ROOT/'result.json').open('x') as f:
            json.dump(result, f, indent=2); f.write('\n')
    if error is not None: raise error

if __name__ == '__main__': main()
