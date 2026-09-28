"""Whole original dimensions, exact K groups and retained bank scope before allocation."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from mixer_qualification import validate, root_rows, canonical_bf16


class MixerQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=ROOT/'evidence/layer-mlp-compile-018/source'
        cls.args=[json.loads((source/n).read_text()) for n in ['joint-stage.json','device-network.json','mixer-setups.json','profiles.json']]
        cls.args.append(json.loads((ROOT/'evidence/layer-joint-reference-003/bank-index.json').read_text()))

    def test_all_original_rows_and_full_k_roots_are_covered(self):
        plan=validate(*self.args)
        expected=[10240,6144,48,48,5120]
        for roots,n in zip(plan['roots'],expected):
            observed=[r+i for root,r in root_rows(roots) for i in range(root['rows'])]
            self.assertEqual(sorted(observed),list(range(n)))
        self.assertEqual(len(plan['device_banks']),3554)
        self.assertEqual(sum(r['words']*4 for r in plan['device_banks']+plan['sdk_banks']),396953216)

    def test_missing_k_peer_or_output_tile_rejected_before_allocation(self):
        for change in ('missing_peer','missing_tile'):
            args=copy.deepcopy(self.args)
            if change=='missing_peer':args[2].pop(1)
            else:args[2][1][1][1]-=1
            with self.assertRaises(ValueError):validate(*args)

    def test_bank_alias_or_missing_bytes_rejected(self):
        for change in ('alias','missing'):
            args=copy.deepcopy(self.args)
            if change=='alias':args[4]['records'][1]['offset']=0
            else:args[4]['records'].pop()
            with self.assertRaises(ValueError):validate(*args)

    def test_numeric_comparison_relaxes_only_zero_sign(self):
        a=np.array([0,0x8000,0x3f80,0xbf80,1,0x8001],np.uint16)
        np.testing.assert_array_equal(canonical_bf16(a),[0,0,0x3f80,0xbf80,1,0x8001])


if __name__=='__main__':unittest.main()
