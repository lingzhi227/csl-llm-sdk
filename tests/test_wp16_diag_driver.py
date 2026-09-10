"""Small synthetic host boundary faults; no SDK, original weights or model calls."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples/wp16/diagnostic'))
from checks import Checks, f64_of_f32, key
from driver import flush_and_stop, upload_values
from qwen38.wp16_diag_layout import sequence
from qwen38.wp16_diag_protocol import descriptor, packet


def handoff_fixture(edge):
    checks = Checks({}, {})
    body = np.zeros(130, dtype=np.uint32)
    body[0], body[-1] = 0xa55a, 0x5aa5
    sender, receiver = edge, 2 if edge < 2 else 3
    phase = 'handoff_' + str(edge)
    origin = ('finalize_projections', 'bf16_storage', sender) if edge < 2 else ('whole_silu_and_product', 'mlp_product', 2)
    checks.observed[key(*origin)] = body.copy()
    for pe in (sender, receiver):
        checks.observed[key(phase, 'wire_data', pe)] = np.array(packet(edge, [0] * 128), dtype=np.uint32)
        checks.observed[key(phase, 'wire_ack', pe)] = np.array([0xa55aa55a, 0x4d4c5041, 1, *descriptor(edge), 0, 0x5aa55aa5], dtype=np.uint32)
        checks.observed[key(phase, 'handoff_latched', pe)] = np.array(descriptor(edge), dtype=np.uint32)
        state = np.zeros(32, dtype=np.uint32)
        state[:5] = [4, 1, 0, 1, edge]
        state[6] = state[11] = state[31] = 1
        state[13:15] = [140, 13]
        state[17:19] = [1, 2]
        state[24:29] = [edge, 0, 128, 0, 128]
        if pe == sender:
            state[5], state[15] = (3, 2) if edge == 2 else (0, 0)
            state[9:11] = [1, 1]
            state[22:24] = [3, 4]
        else:
            state[5] = [1, 3, 4][edge]
            state[15] = 2 if edge == 1 else 1
            state[7:9] = [1, 1]
            state[19:22] = [3, 4, 5]
            state[23] = 6
        state[16] = state[23]
        checks.observed[key(phase, 'handoff_state', pe)] = state
    checks.observed[key(phase, ('mlp_gate', 'mlp_up', 'mlp_product')[edge], receiver)] = body.copy()
    mlp = np.zeros(16, dtype=np.uint32)
    mlp[:5] = [1, 1, 0, 1, [1, 3, 4][edge]]
    if edge == 2:
        mlp[7] = 128
    checks.observed[key(phase, 'mlp_state', receiver)] = mlp
    return checks


class DriverBoundaryTests(unittest.TestCase):
    def test_flush_failure_preserves_primary_and_never_bypasses_stop(self):
        class EvidenceFailure:
            files = {}; records = 0
            def flush(self):
                raise OSError('synthetic disk failure')
            def event(self, kind):
                raise OSError('synthetic journal failure')
            def write(self, *args):
                raise OSError('synthetic summary failure')
        class Runtime:
            calls = 0
            def stop(self):
                self.calls += 1
        original = ValueError('original numerical failure')
        for primary in (None, original):
            runner = Runtime(); report = {}
            error, stopped = flush_and_stop(EvidenceFailure(), runner, report, primary)
            self.assertEqual(runner.calls, 1)
            self.assertTrue(stopped)
            if primary is original:
                self.assertIs(error, original)
            else:
                self.assertIsInstance(error, OSError)
            self.assertIn('pre_stop_flush_error', report)

    def test_exact_finite_FP32_bits_including_subnormal_and_signed_zero(self):
        words = np.array([0, 0x80000000, 1, 0x80000001, 0x3f800000], dtype=np.uint32)
        decoded = f64_of_f32(words.view(np.float32))
        np.testing.assert_array_equal(decoded, [0., -0., 2.**-149, -2.**-149, 1.])
        self.assertTrue(np.signbit(decoded[1]))
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            f64_of_f32(np.array([0x7f800000], dtype=np.uint32).view(np.float32))

    def test_nonlinear_cannot_accept_mutated_received_operands(self):
        for changed_name in ('mlp_gate', 'mlp_up'):
            checks = Checks({}, {})
            for edge, name in enumerate(('mlp_gate', 'mlp_up')):
                body = np.zeros(130, dtype=np.uint32)
                body[0], body[-1] = 0xa55a, 0x5aa5
                checks.observed[key('handoff_' + str(edge), name, 2)] = body
                checks.observed[key('whole_silu_and_product', name, 2)] = body.copy()
            checks.observed[key('whole_silu_and_product', changed_name, 2)][1] ^= 1
            with self.assertRaisesRegex(ValueError, 'Exact received operand retained'):
                checks.nonlinear()

    def test_join_metadata_faults_fail_even_with_exact_numeric_payload(self):
        for edge in range(3):
            handoff_fixture(edge).handoff(edge)
            sender, receiver = edge, 2 if edge < 2 else 3
            for pe, slot in ((sender, 7), (sender, 5), (sender, 15), (sender, 16), (receiver, 9), (receiver, 22)):
                checks = handoff_fixture(edge)
                checks.observed[key('handoff_' + str(edge), 'handoff_state', pe)][slot] ^= 1
                with self.assertRaises(ValueError):
                    checks.handoff(edge)

    def test_last_original_input_tile_packing_and_consumer_injection_rejection(self):
        columns = np.arange(5120, dtype=np.uint16)
        matrix = np.broadcast_to(columns, (128, 5120))
        weights = dict(gate_proj=matrix)
        item = next(x for x in sequence() if x['kind'] == 'copy' and x['direction'] == 'h2d' and
                    x['name'] == 'weights_storage' and x['pe'] == 0 and x['tile'] == 45)
        packed = upload_values(item, weights, columns)
        np.testing.assert_array_equal(packed[1:-1].reshape(112, 128)[:80], matrix[:, 5040:].T)
        self.assertTrue(np.all(packed[1:-1].reshape(112, 128)[80:] == 0x3e80))
        item = dict(item, name='input_storage', count=114)
        hidden = upload_values(item, weights, columns)
        np.testing.assert_array_equal(hidden.view(np.uint32)[1:81], columns[5040:].astype(np.uint32) << 16)
        self.assertTrue(np.all(hidden[81:-1] == 7.))
        with self.assertRaisesRegex(ValueError, 'No host numerical consumer input'):
            upload_values(dict(item, pe=3), weights, columns)


if __name__ == '__main__':
    unittest.main()
