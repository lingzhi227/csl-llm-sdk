"""Independent tiling/route acceptance and fault rejection for the P18 component."""
import copy
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.retiled_matrix import RetiledMatrixPlan
from tools.audit_retiled_matrix_plan import audit

class RetiledMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e=Path(__file__).resolve().parents[1]/'evidence'
        cls.base=json.loads((e/'model-atlas-001.json').read_text())
        cls.overlay=json.loads((e/'columnar-plan-001.json').read_text())
        cls.plan=RetiledMatrixPlan(e/'model-atlas-001.json',e/'columnar-plan-001.json')
        cls.document=cls.plan.document()

    def test_complete_partition_and_route_graph(self):
        receipt=audit(self.base,self.overlay,self.document)
        self.assertTrue(receipt['passed'])
        self.assertEqual(receipt['contraction_edges'],954)
        self.assertEqual(receipt['gather_edges'],5)
        self.assertEqual(receipt['input_recipients'],960)
        self.assertEqual(receipt['input_words'],2720)

    def test_wrong_tile_or_bank_identity_rejected(self):
        for field in ['row_start','column_start','target_slot','fp8_slots','eligible_rank']:
            d=copy.deepcopy(self.document);d['workers'][159][field]+=1
            with self.assertRaises(AssertionError):audit(self.base,self.overlay,d)

    def test_bad_route_and_sdk_color_rejected(self):
        for mode in ['port','sdk','duplicate']:
            d=copy.deepcopy(self.document)
            if mode=='port':d['routes'][0]['tx']=['SOUTH']
            if mode=='sdk':d['routes'][0]['color']=20
            if mode=='duplicate':d['routes'].append(d['routes'][0])
            with self.assertRaises((AssertionError,KeyError)):audit(self.base,self.overlay,d)

    def test_gather_order_is_all_48_rows(self):
        values={}
        for g in range(5,-1,-1):
            profile=self.plan.gather(g);value=list(range(g*8,g*8+8))
            for s in range(profile['steps']):value+=values[g+2**s]
            self.assertEqual(len(value),profile['words']);values[g]=value
        self.assertEqual(values[0],list(range(48)))
        self.assertEqual({r['color'] for r in self.document['routes']},set(range(2,20)))
