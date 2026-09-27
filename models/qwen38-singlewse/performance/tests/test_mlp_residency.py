import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.model_subgraph import mlp_subgraph
from spatial.mlp_regions import MlpRegions
from spatial.mlp_residency import MlpResidency
from tools.audit_mlp_residency import audit


class MlpResidencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.graph=json.loads((root/'configs/model-graph.json').read_text())
        cls.tensors=json.loads((root/'configs/tensors.json').read_text())['tensors']
        cls.r=MlpResidency(MlpRegions(mlp_subgraph(cls.graph,cls.tensors)),cls.graph,cls.tensors)

    def test_complete_original_inventory_state_and_profile_intervals(self):
        a=audit(self.r,self.graph,self.tensors)
        self.assertEqual(a['original_tensors'],1251)
        self.assertEqual(a['matrix_tiles'],105052160)
        self.assertEqual(a['embedding_helpers'],5800)
        self.assertEqual(a['maximum_matrix_state_and_allocated_aux_payload'],35232)
        self.assertFalse(a['compiled_all_role_sram_admitted'])

    def test_fixed_spatial_workers_hold_all64_mlp_layers_without_slot_aliases(self):
        gate_slots=set();up_slots=set();g,p,k=7,13,23
        for layer in range(64):
            prefix=f'model.language_model.layers.{layer}.mlp.'
            gate=self.r.address(prefix+'gate_proj.weight',(g*64+p)*40+k)
            down=self.r.address(prefix+'down_proj.weight',(k*64+p)*136+g)
            up=self.r.address(prefix+'up_proj.weight',(g*64+p)*40+k)
            self.assertEqual(gate['xy'],down['xy']);self.assertNotEqual(up['xy'],gate['xy'])
            gate_slots.update([gate['slot'],down['slot']]);up_slots.add(up['slot'])
            self.assertEqual(gate['original_scale_index'],[g,k]);self.assertEqual(down['original_scale_index'],[k,g])
        self.assertEqual(gate_slots,set(range(128)));self.assertEqual(up_slots,set(range(64)))

    def test_embedding_helpers_reject_hidden_fp8_compute_obligations(self):
        i=self.r.helpers[0];old=self.r.fp8[i]
        try:
            self.r.fp8[i]=1
            with self.assertRaises(AssertionError):audit(self.r,self.graph,self.tensors)
        finally:self.r.fp8[i]=old

    def test_lost_full_head_rejected(self):
        saved=self.r.matrices.pop('lm_head.weight')
        try:
            with self.assertRaises(AssertionError):audit(self.r,self.graph,self.tensors)
        finally:self.r.matrices['lm_head.weight']=saved

    def test_auxiliary_bounds_and_nearby_norm_ownership(self):
        for page in [0,self.r.used_aux_pages-1]:
            a=self.r.auxiliary_address(page);self.assertLessEqual(a['byte_offset']+128,35256)
        for page in [-1,self.r.used_aux_pages]:
            with self.assertRaises(ValueError):self.r.auxiliary_address(page)
        a=self.r.norm_address('model.language_model.layers.63.post_attention_layernorm.weight',39)
        self.assertEqual(a['xy'],[0,39]);self.assertEqual(a['rows'],[4992,5120]);self.assertLessEqual(a['byte_offset']+256,33024)

    def test_context_mismatch_rejected(self):
        wrong=dict(self.graph,context=192)
        with self.assertRaises(ValueError):MlpResidency(self.r.flow,wrong,self.tensors)
