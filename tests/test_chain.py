from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.chain_numerics import reference,check,validate
from qwen38.numerics import inputs


class ChainTests(unittest.TestCase):
    def test_zero_and_epsilon_remain_finite(self):
        result=reference([1.0]*(128*112),inputs()[3][1])
        self.assertGreater(result['q'],0)
        self.assertEqual(result['normalized'],[0.0]*128)
        self.assertTrue(check([0.0]*128,result['normalized'],result['normalized_bound'],True)['passed'])
        self.assertFalse(check([1e-40]*128,result['normalized'],result['normalized_bound'],True)['passed'])

    def test_both_join_orders_and_premature_commit_rejected(self):
        sender=[1,4,1,0,0,0,4,3,0,0,1,0,0,2]
        local=[1,6,1,3,4,5,6,0,1,1,1,0,2,0]
        remote=[1,6,3,2,4,5,6,0,1,1,1,0,1,0]
        self.assertTrue(validate([local,sender],0,1))
        self.assertTrue(validate([remote,sender],1,1))
        self.assertFalse(validate([local,sender],1,1))
        premature=remote.copy();premature[4]=2
        self.assertFalse(validate([premature,sender],1,1))
        duplicate=local.copy();duplicate[10]=2
        self.assertFalse(validate([duplicate,sender],0,1))

    def test_missing_duplicate_commit_and_early_unblock(self):
        sender=[1,4,1,0,0,0,4,3,0,0,1,0,0,2]
        root=[1,6,1,3,4,5,6,0,1,1,1,0,2,0]
        for slot,value in ((9,0),(9,2),(6,2),(8,0),(8,2)):
            bad=root.copy();bad[slot]=value
            self.assertFalse(validate([bad,sender],0,1))
