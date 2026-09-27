"""Boundary regressions for complete tensor-owner input pairing."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.columnar import consumer,packet_groups,source,segments
from spatial.filtered_bank import FilteredBankPlan

class ColumnarTests(unittest.TestCase):
    def test_complete_input_partition_and_last_wide_group(self):
        for k in [40,48,136]:
            groups=packet_groups(k)
            self.assertEqual(sorted(v for g in groups for v in g['packet_blocks']),list(range(k)))
        self.assertEqual(packet_groups(136)[-1]['packet_blocks'],[66,69,67,68])
        self.assertEqual(source(136,68,0)['horizontal_window'],dict(offset_words=130,pass_words=130,period_words=260))

    def test_partial_last_row_and_removed_wide_border(self):
        self.assertIsNone(consumer(40,1147,119))
        self.assertIsNotNone(consumer(40,1147,120))
        for row,x in [(1146,1),(0,0),(0,749)]:self.assertIsNone(consumer(136,row,x))
        for k,row,x in [(40,1147,120),(48,1147,749),(136,1145,1),(136,0,748)]:
            actual=consumer(k,row,x);src=source(k,x,actual['lane'])
            self.assertEqual(src['blocks'][actual['vertical_window']['offset_words']//65],actual['block'])
        with self.assertRaises(ValueError):source(136,749,0)
        with self.assertRaises(ValueError):source(48,0,4)

    def test_aligned_slot_wrap_and_partial_group_rejection(self):
        parts=segments(80,280,120,40)
        self.assertEqual([(s['class_start'],s['count'],s['slot']) for s in parts],[(80,40,0),(0,120,1),(0,120,2)])
        for args in [(1,40,120,40),(0,39,120,40),(80,80,121,40)]:
            with self.assertRaises(ValueError):segments(*args)

    def test_probe_exercises_both_maximum_admitted_storage_profiles(self):
        workers=FilteredBankPlan().workers()
        self.assertEqual(len({(w['x'],w['y']) for w in workers}),12)
        profiles={(w['fp8_slots'],w['bf16_slots']) for w in workers}
        self.assertEqual(profiles,{(110,13),(111,12)})
        self.assertEqual(max(f*260+b*512 for f,b in profiles),35256)
