"""Original graph ownership and independent RMS checks; not device acceptance."""
import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'reference')]
from spatial.layer_mlp_network import build_mlp_network, worker_profiles, audit_mlp_network
from norm_oracle import residual_rms
from mlp_oracle import bf16_bits, bf16_float
from remap_banks import bank_segments
sys.path.insert(0,str(ROOT/'tools'))
from stage_layer_mlp_compile import build


class NormBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan=json.loads((ROOT/'evidence/layer-native-schedule-002/layer-schedule.json').read_text())
        cls.stage=next(s for s in plan['stages'] if s['id']=='layer_00')
        cls.routes=json.loads((ROOT/'evidence/layer-fused-routes-003/routes-78x67.json').read_text())
        cls.old=build_mlp_network(cls.stage,cls.routes,True)
        cls.new=build_mlp_network(cls.stage,cls.routes,True,True)

    def test_original_native_banks_and_math_ownership_preserved(self):
        old=worker_profiles(self.stage,self.routes,self.old);new=worker_profiles(self.stage,self.routes,self.new)
        for a,b in zip(old,new):
            for field in ['pe','arm','original_matrix_bytes','live_auxiliary_bytes','native_input_slices']:
                self.assertEqual(a[field],b[field])
            for field in ['rows','columns','branches','bank_words','left_color','right_color','subtree_k']:
                self.assertEqual(a['parameters'][field],b['parameters'][field])
        self.assertEqual(self.old['input_bindings'],self.new['input_bindings'])
        self.assertEqual(self.old['distribution_packets'],self.new['distribution_packets'])
        self.assertEqual(sum(w['parameters'].get('input_quantizer',False) for w in new),0)
        self.assertEqual(sum(w['parameters'].get('fusion_actor',False) for w in new),33)
        self.assertEqual(len(self.new['senders']),83)

    def test_bad_norm_data_and_reduction_routes_rejected(self):
        for color in (4,5,6,7,8):
            bad=copy.deepcopy(self.new)
            # Mutate the first norm-specific route of this color inside mixer.
            r=next(r for r in bad['routes'] if r['color']==color and r['pe'][1]==144 and r['pe'][0]<103)
            r['tx']=[]
            with self.assertRaises(ValueError):audit_mlp_network(self.stage,bad)
        bad=copy.deepcopy(self.new);bad['grant_schedule'][-1]['first_row']=0
        with self.assertRaises(ValueError):audit_mlp_network(self.stage,bad)

    def test_bf16_residual_boundary_and_independent_full_width_rms(self):
        rng=np.random.default_rng(3138)
        for mode in ['zero','random','cancel','wide']:
            x=rng.normal(size=5120).astype(np.float32);y=rng.normal(size=5120).astype(np.float32)
            if mode=='zero':x*=0;y*=0
            if mode=='cancel':y=-x
            if mode=='wide':x*=np.exp2(rng.integers(-20,21,size=5120)).astype(np.float32)
            a,b,g=map(bf16_bits,[x,y,rng.normal(0,.1,size=5120).astype(np.float32)])
            r=residual_rms(a,b,g)
            truth=bf16_bits(bf16_float(a).astype(np.float64)+bf16_float(b).astype(np.float64))
            np.testing.assert_array_equal(r['residual'],truth)
            self.assertTrue(np.all(np.abs(r['fp32'].astype(np.float64)-r['fp64'])<=r['bound']))
            if mode in ['zero','cancel']:self.assertFalse(np.any(bf16_float(r['normalized'])))

    def test_foreign_shape_and_unbounded_fixture_rejected(self):
        zeros=np.zeros(5120,np.uint16)
        with self.assertRaises(ValueError):residual_rms(zeros[:128],zeros,zeros)
        with self.assertRaises(ValueError):residual_rms(zeros.astype(np.uint32),zeros,zeros)
        with self.assertRaises(ValueError):residual_rms(bf16_bits(np.full(5120,2**40,np.float32)),zeros,zeros)

    def test_actual_full_bank_remap_conserves_every_original_word(self):
        files,_,_,_=build('layer_00',shared_inputs=True,norm_bridge=True)
        original=json.loads((ROOT/'evidence/layer-mlp-reference-001/bank-index.json').read_text())['records']
        profiles=json.loads(files['profiles.json'])['profiles'];pages=json.loads(files['state-page-remap.json'])['pages']
        plan=bank_segments(original,profiles,pages)
        self.assertEqual(plan['total_words'],99238304)
        self.assertEqual(len(pages),800)
        self.assertEqual({p['tensor'] for p in pages},{'recurrent.0'})
        self.assertTrue(all(p['source'][0]==p['destination'][0] and p['source'][1]-p['destination'][1]==2 for p in pages))
        # Lost, duplicated, misplaced and matrix-prefix transfers all fail the
        # complete source+destination partition proof, before model bytes move.
        for action in ['lost','duplicate','destination','matrix']:
            bad=copy.deepcopy(pages)
            if action=='lost':bad.pop()
            elif action=='duplicate':bad[1]=copy.deepcopy(bad[0])
            elif action=='destination':bad[0]['destination_byte_offset']+=128
            else:bad[0]['source_byte_offset']=0
            with self.assertRaises(ValueError):bank_segments(original,profiles,bad)


if __name__=='__main__':unittest.main()
