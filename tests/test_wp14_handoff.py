"""Host framing/completion invariants, explicitly not SDK malformed recovery."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.projection_handoff import ConsumerModel, descriptor, data_frame, acknowledgement


class HandoffTests(unittest.TestCase):
    def test_both_completion_orders_and_noncontiguous_slabs(self):
        model = ConsumerModel()
        expected = []
        for rank in range(4):
            control = descriptor(rank)
            self.assertEqual(control[3], [0, 1, 4, 5][rank])
            words = [rank*128+i for i in range(128)]; expected += words
            model.arm(control)
            if rank % 2 == 0:
                model.command()
            self.assertEqual(model.receive(data_frame(control, words)), acknowledgement(control))
            self.assertEqual(model.unblocks, rank)
            model.sent_ack()
            if rank % 2:
                self.assertEqual(model.unblocks, rank)
                model.command()
            self.assertEqual(model.unblocks, rank+1)
        self.assertEqual(model.preprocessing_input(), expected)

    def test_bad_frame_cannot_partially_mutate_operand(self):
        for kind in ('slab', 'destination', 'high16', 'truncated', 'generation'):
            model = ConsumerModel(); control = descriptor(2); model.arm(control)
            frame = data_frame(control, [0x8000]*128)
            if kind == 'slab': frame[5] = 2
            if kind == 'destination': frame[9] = 500
            if kind == 'high16': frame[-1] = 65536
            if kind == 'truncated': frame.pop()
            if kind == 'generation': frame[2] += 1
            with self.assertRaises(ValueError): model.receive(frame)
            self.assertEqual(model.mask, 0)
            self.assertEqual(model.words, [None]*512)

    def test_missing_duplicate_stale_and_ack_before_commit(self):
        model = ConsumerModel()
        with self.assertRaises(ValueError): model.preprocessing_input()
        with self.assertRaises(ValueError): model.arm(descriptor(0, 2))
        c = descriptor(0); model.arm(c)
        with self.assertRaises(ValueError): model.sent_ack()
        model.receive(data_frame(c, [0]*128))
        with self.assertRaises(ValueError): model.receive(data_frame(c, [0]*128))
        model.sent_ack(); model.command()
        with self.assertRaises(ValueError): model.arm(c)
        with self.assertRaises(ValueError): model.preprocessing_input()
