"""Tiny offline frame/schedule checks; no SDK, original parameters or inference."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.wp16_diag_layout import budget, sequence
from qwen38.wp16_diag_protocol import Receiver, descriptor, packet, verify_ack


class DiagnosticProtocolTests(unittest.TestCase):
    def test_both_abstract_command_receipt_joins(self):
        for command_first in (True, False):
            receiver = Receiver(2)
            for edge in (0, 1):
                control = list(descriptor(edge))
                receiver.arm(control)
                control[0] = 99  # The mutable host descriptor must not affect the latch.
                if command_first:
                    receiver.command()
                ack = receiver.receive(packet(edge, [0x3f80] * 128))
                self.assertFalse(receiver.command_complete)
                verify_ack(edge, ack)
                receiver.acknowledge_sent()
                if not command_first:
                    self.assertFalse(receiver.command_complete)
                    receiver.command()
                self.assertTrue(receiver.command_complete)
            self.assertTrue(receiver.ready_for_compute)
            before = dict(receiver.committed)
            with self.assertRaises(ValueError):
                receiver.arm(descriptor(1))
            self.assertEqual(before, receiver.committed)

    def test_invalid_full_frames_never_publish_partial_payload(self):
        original = packet(2, [0x3f80] * 128)
        mutations = [original[:-1], list(original), list(original), list(original)]
        mutations[1][3] = 9
        mutations[2][140] = 0x10000
        mutations[3][13] = 0x7f80
        for frame in mutations:
            receiver = Receiver(3)
            receiver.arm(descriptor(2))
            receiver.command()
            ack = receiver.receive(frame)
            receiver.acknowledge_sent()
            self.assertEqual(receiver.committed, {})
            self.assertFalse(receiver.ready_for_compute)
            with self.assertRaises(ValueError):
                verify_ack(2, ack)

    def test_wrong_descriptor_rejected_before_arm_and_payload_mutation(self):
        receiver = Receiver(2)
        for control in (descriptor(1), (2, *descriptor(0)[1:]), (True, *descriptor(0)[1:])):
            with self.assertRaises(ValueError):
                receiver.arm(control)
            self.assertIsNone(receiver.latched)
            self.assertEqual(receiver.committed, {})

    def test_tile_observation_group_precedes_overwrite_and_consumers_have_no_host_operand(self):
        pending = set()
        for item in sequence():
            if item['kind'] != 'copy':
                continue
            if item['name'] == 'weights_storage':
                pe = item['pe']
                if item['direction'] == 'h2d':
                    self.assertNotIn(pe, pending)
                    pending.add(pe)
            if item['name'] == 'tile_guard_snapshot' and item['direction'] == 'd2h':
                self.assertIn(item['pe'],pending)
                pending.remove(item['pe'])
            if item['direction'] == 'h2d' and item['pe'] >= 2:
                self.assertIn(item['name'], ('weights_storage', 'control', 'handoff_control', 'release_control'))
        self.assertFalse(pending)
        self.assertEqual(budget()['weight_copies_by_direction'], {'h2d': 94, 'd2h': 9})
        self.assertLessEqual(budget()['maximum_host_buffer_bytes'], 65536)


if __name__ == '__main__':
    unittest.main()
