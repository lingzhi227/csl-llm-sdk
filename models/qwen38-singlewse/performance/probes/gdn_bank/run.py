"""Complete dependent recurrence, exact block oracle, original FP64 gate and bank retention."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from backend import runtime


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--physical', action='store_true'); args = parser.parse_args()
    meta = json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest() == meta['fixture_sha256']
    with np.load('fixture.npz', allow_pickle=False) as f:
        data = {n: f[n] for n in f.files}
    records = []
    with runtime(args.physical) as (runner, dtype, order):
        ids = {n: runner.get_id(n) for n in ['weights', 'scale', 'state', 'dot_input', 'dot_output', 'audit', 'requests', 'results', 'ticks']}
        options = dict(streaming=False, order=order.ROW_MAJOR, nonblock=False)
        def upload(name, value, x=0, y=1, bits=32):
            value = np.ascontiguousarray(value); h, w, n = value.shape
            runner.memcpy_h2d(ids[name], value.reshape(-1), x, y, w, h, n,
                              data_type=dtype.MEMCPY_16BIT if bits == 16 else dtype.MEMCPY_32BIT, **options)
        def read(name, count, x=0, y=1, w=4, h=4, bits=32):
            out = np.zeros(w * h * count, np.uint32)
            runner.memcpy_d2h(out, ids[name], x, y, w, h, count,
                              data_type=dtype.MEMCPY_16BIT if bits == 16 else dtype.MEMCPY_32BIT, **options)
            return out.reshape(h, w, count)
        def state():
            return read('state', 1024).view(np.float32).reshape(4, 4, 32, 32).transpose(1, 2, 0, 3).reshape(128, 128)
        def canonical(v):
            return np.where(v == 0, np.float32(0), v).astype(np.float32).view(np.uint32)
        def dots(expected_state):
            print(json.dumps(dict(phase='native_fp8_dot_checks')), flush=True)
            for i, slot in enumerate(meta['dot_slots']):
                upload('dot_input', data['dot_input'][i]); runner.launch('dot', np.uint16(slot), nonblock=False)
                got = read('dot_output', 2).view(np.float32)
                np.testing.assert_array_equal(canonical(got), canonical(data['dot_output'][i]))
                assert np.all(np.abs(got.astype(np.float64) - data['dot_exact'][i]) <= data['dot_bounds'][i])
            np.testing.assert_array_equal(state().view(np.uint32), expected_state.view(np.uint32))
        started = time.perf_counter()
        upload('weights', np.ascontiguousarray(data['weights']).view(np.uint32)); upload('scale', data['scales'])
        initialization = time.perf_counter() - started
        print(json.dumps(dict(phase='banks_initialized', seconds=initialization)), flush=True)
        lengths = [meta['positions'] if args.physical else meta['simulation_positions'],
                   meta['physical_replay'] if args.physical else meta['simulation_replay']]
        for replay, length in enumerate(lengths):
            print(json.dumps(dict(phase='reset', replay=replay, positions=length)), flush=True)
            runner.launch('reset', nonblock=False); assert not np.any(read('audit', 8, y=0, h=5))
            if replay == 0:
                dots(np.zeros((128, 128), np.float32))
            else:
                assert not np.any(state())
            for first in range(0, length, 8):
                count = min(8, length - first); last = first + count - 1
                requests = np.zeros((8, 387), np.uint32)
                requests[:count, 0] = np.arange(first, first + count)
                requests[:count, 1:] = data['packet'][first:first + count].view(np.uint32)
                upload('requests', requests.reshape(1, 1, -1), y=0)
                started = time.perf_counter(); runner.launch('initialize', np.uint16(count), np.uint16(first), nonblock=False)
                audit = read('audit', 8, y=0, h=5); assert not np.any(audit[:, :, 6]); assert np.all(audit[1:, :, 1] == first)
                arm_seconds = time.perf_counter() - started
                print(json.dumps(dict(phase='autonomous_batch', replay=replay, first=first, count=count)), flush=True)
                started = time.perf_counter(); runner.launch('start', nonblock=False)
                result = read('results', count * 4 * 35, y=0, w=1, h=1).reshape(count, 4, 35)
                host_seconds = time.perf_counter() - started
                deadline = time.monotonic() + 5
                while True:
                    audit = read('audit', 8, y=0, h=5)
                    if np.all(audit[:, :, 6] == 1):
                        break
                    if time.monotonic() > deadline:
                        raise AssertionError('Terminal callbacks not complete')
                expected = np.zeros_like(audit)
                expected[:, :, 0] = first // 8 + 1; expected[:, :, 6] = 1
                expected[1:, :, 1] = last + 1; expected[1:, :, 2] = last + 1
                expected[1:, :, 3] = 2 * (last + 1); expected[1:, 1:, 4] = last + 1
                expected[1:, :, 5] = 2 * (last + 1); expected[1:, :, 7] = 3 if replay == 0 else 0
                expected[0, 0, 1] = last + 1; expected[0, 0, 3] = 4 * (last + 1); expected[0, 0, 5] = last + 1
                np.testing.assert_array_equal(audit, expected)
                output = np.ascontiguousarray(result[:, :, :32]).view(np.float32).reshape(count, 128)
                current = state()
                np.testing.assert_array_equal(result[:, :, 32], np.full((count, 4), 4))
                np.testing.assert_array_equal(result[:, :, 33], np.broadcast_to(np.arange(first, first + count)[:, None], (count, 4)))
                np.testing.assert_array_equal(result[:, :, 34], np.ones((count, 4)))
                np.testing.assert_array_equal(canonical(output), canonical(data['expected_output'][first:first + count]))
                np.testing.assert_array_equal(canonical(current), canonical(data['expected_state'][last]))
                se = np.abs(current.astype(np.float64) - data['fp64_state'][last])
                oe = np.abs(output.astype(np.float64) - data['fp64_output'][first:first + count])
                assert np.isfinite(output).all() and np.isfinite(current).all()
                assert np.all(se <= data['state_bound'][last]) and np.all(oe <= data['output_bound'][first:first + count])
                ticks = read('ticks', count * 6, y=0, w=1, h=1, bits=16).reshape(count, 6)
                starts = [sum(int(t[k]) << (16 * k) for k in range(3)) for t in ticks]
                ends = [sum(int(t[k + 3]) << (16 * k) for k in range(3)) for t in ticks]
                cycles = [(b - a) % (1 << 48) for a, b in zip(starts, ends)]
                gaps = [(starts[k + 1] - ends[k]) % (1 << 48) for k in range(count - 1)]
                total = (ends[-1] - starts[0]) % (1 << 48); assert total == sum(cycles) + sum(gaps)
                record = dict(replay=bool(replay), first=first, count=count, request_to_full_state_output_cycles=cycles,
                              inter_step_cycles=gaps, autonomous_batch_cycles=total, max_state_fp64_error=float(se.max()),
                              max_output_fp64_error=float(oe.max()), host_arm_seconds=arm_seconds, host_batch_result_seconds=host_seconds)
                records.append(record); print(json.dumps(record), flush=True)
                np.savez('actual-%d-%03d.npz' % (replay, first), state=current, output=output, audit=audit, ticks=ticks)
        dots(current)
        assert np.all(read('audit', 8)[..., 7] == 3)
        print(json.dumps(dict(phase='full_bank_retention')), flush=True)
        np.testing.assert_array_equal(read('weights', 111 * 64), np.ascontiguousarray(data['weights']).view(np.uint32))
        print(json.dumps(dict(phase='packed_weights_retained')), flush=True)
        np.testing.assert_array_equal(read('scale', 111), data['scales'].view(np.uint32))
    result = dict(passed=True, normal_stop=True, physical=args.physical, full_model=False, application=[4, 5],
                  fixture_sha256=meta['fixture_sha256'], scope=meta['scope'], batches=records,
                  original_weight_uploads=1, all_original_banks_retained=True, state_unchanged_by_native_fp8_dots=True,
                  native_fp8_dot_values_checked=2 * 3 * 16 * 2, state_elements_per_terminal_check=16384,
                  all_outputs_and_terminal_states_exact_block_order_fp32=True, unchanged_fp64_bounds_passed=True,
                  host_initialization_seconds=initialization, clock_hz=None,
                  timing_note='Same-controller request issue through all four complete tagged output shards; send callback and terminal worker callback audit are separate. Eight state-dependent recurrences per batch with no host access inside. Preprocessed synthetic neural operands, no convolution/gates/norm or model/token rate.')
    Path('result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, physical=args.physical, positions=sum(lengths))), flush=True)


if __name__ == '__main__':
    main()
