import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.pipeline import PipelineConfig,place_pipeline,audit_pipeline,bank_profile,local_tile_owner,auxiliary_owner
from spatial.pipeline_lowering import lower_pipeline,audit_boundaries
from spatial.pipeline_protocol import Tag,StageMailbox,FeedbackAdmission


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.graph=json.loads((root/'configs/model-graph.json').read_text())
        cls.tensors=json.loads((root/'configs/tensors.json').read_text())['tensors']
        cls.plan=place_pipeline(cls.graph,cls.tensors)

    def test_whole_original_model_disjoint_and_two_requests(self):
        a=audit_pipeline(self.plan,self.graph,self.tensors)
        self.assertEqual((a['original_tensors'],a['original_operations'],a['layers']),(1251,1172,64))
        self.assertEqual(a['original_weight_bytes'],29468003328)
        self.assertEqual(a['state_bytes_per_request'],48*48*128*128*4+48*3*10240*2+16*96*4*256*2*2)
        self.assertTrue(a['admitted']);self.assertFalse(a['compiled_sram_qualified'])
        self.assertEqual(self.plan['config']['concurrency'],2)
        # Same two-dimensional neighborhood never stores another layer's weights.
        for layer,stage in enumerate(self.plan['stages'][1:-1]):
            self.assertEqual(stage['layer'],layer)
            self.assertTrue(all(f'.layers.{layer}.' in w for r in stage['regions'] for w in r['weights']))

    def test_overcommit_and_damaged_coverage_are_rejected(self):
        bad=copy.deepcopy(self.plan);bad['stages'][2]['rect']=bad['stages'][1]['rect']
        with self.assertRaises(ValueError):audit_pipeline(bad,self.graph,self.tensors)
        bad=copy.deepcopy(self.plan);bad['stages'][1]['regions'][0]['weights'].pop()
        with self.assertRaises(ValueError):audit_pipeline(bad,self.graph,self.tensors)
        bad=copy.deepcopy(self.plan);bad['stages'][1]['regions'][0]['matrices'].pop()
        with self.assertRaises(ValueError):audit_pipeline(bad,self.graph,self.tensors)
        bad=copy.deepcopy(self.plan);bad['stages'][1]['regions'][0]['auxiliary'][0]['page_start']+=1
        with self.assertRaises(ValueError):audit_pipeline(bad,self.graph,self.tensors)
        large=place_pipeline(self.graph,self.tensors,PipelineConfig(concurrency=64))
        self.assertFalse(audit_pipeline(large,self.graph,self.tensors)['admitted'])
        with self.assertRaises(ValueError):lower_pipeline(large,self.graph,self.tensors)

    def test_bank_census_against_enumerated_assignment(self):
        for fp,bf,pes in [(10,21,7),(25,13,8),(1,1,5),(23,26,11)]:
            p=bank_profile(fp,bf,0,pes,4096);actual=[[0,0] for _ in range(pes)]
            for i in range(fp):actual[i%pes][0]+=1
            for i in range(bf):actual[(i+fp%pes)%pes][1]+=1
            for c in p['classes']:
                for rank in range(c['rank_start'],c['rank_end']):
                    self.assertEqual(actual[rank],[c['fp8_slots'],c['bf16_slots']])
                    self.assertEqual(c['auxiliary_pages'],(4096-actual[rank][0]*260-actual[rank][1]*256)//128)

    def test_original_tile_addresses_and_request_state_do_not_alias(self):
        for stage in self.plan['stages']:
            for r in stage['regions']:
                x,y,w,h=r['rect'];n=w*h
                for m in r['matrices']:
                    for tile in {0,m['tiles']-1,min(m['tiles']-1,n-1),min(m['tiles']-1,n)}:
                        owner=local_tile_owner(r,m,tile);px,py=owner['pe']
                        self.assertTrue(x<=px<x+w and y<=py<y+h)
                        self.assertLessEqual(owner['byte_offset']+owner['bytes'],35256)
                for a in r['auxiliary']:
                    if a['kind']!='request_state':continue
                    size=a['bytes_per_request'];self.assertEqual(size%128,0)
                    starts=[auxiliary_owner(r,a['page_start']+request*size//128) for request in range(2)]
                    self.assertNotEqual(starts[0],starts[1])
                    last=auxiliary_owner(r,a['page_start']+a['pages']-1)
                    self.assertLessEqual(last['byte_offset']+last['bytes'],35256)

    def test_fused_edges_have_real_adjacent_ports_and_keep_precision_barriers(self):
        ir=lower_pipeline(self.plan,self.graph,self.tensors);a=audit_boundaries(ir)
        self.assertEqual(a['adjacent_stage_interfaces'],65)
        self.assertFalse(a['complete_fabric_routes_admitted'])
        self.assertEqual(sum(len(k['fusions']) for k in ir['kernels']),192)
        self.assertIn('All136',ir['kernels'][1]['fusions'][-1]['readiness'])
        bad=copy.deepcopy(ir);bad['stage_boundaries'][0]['boundary_ports'][0]['destination']=[0,0]
        with self.assertRaises(ValueError):audit_boundaries(bad)


class PipelineProtocolTests(unittest.TestCase):
    def ready(self,box,lease,tag):
        for chunk in reversed(range(box.chunks)):box.arrive(lease,tag,chunk)
        box.commit_state(lease,tag)

    def test_backpressure_and_both_completion_orders(self):
        box=StageMailbox(3);a=Tag(0,0,0);b=Tag(1,0,0);c=Tag(2,0,0)
        x=box.grant(a);y=box.grant(b);self.assertIsNone(box.grant(c))
        self.ready(box,x,a);self.ready(box,y,b)
        box.complete(x,a,send=True);self.assertIsNone(box.grant(c))
        box.complete(y,b,credit=True);self.assertIsNone(box.grant(c))
        box.complete(x,a,credit=True);z=box.grant(c);self.assertIsNotNone(z)
        box.complete(y,b,send=True)
        with self.assertRaises(ValueError):box.arrive(x,a,0)

    def test_partial_duplicate_wrong_request_and_state_update_rejected(self):
        box=StageMailbox(2);a=Tag(0,0,0);x=box.grant(a)
        with self.assertRaises(ValueError):box.commit_state(x,a)
        with self.assertRaises(ValueError):box.arrive(x,Tag(1,0,0),0)
        box.arrive(x,a,0)
        with self.assertRaises(ValueError):box.arrive(x,a,0)
        for chunk in range(1,40):box.arrive(x,a,chunk)
        box.commit_state(x,a)
        with self.assertRaises(ValueError):box.commit_state(x,a)

    def test_warm_reset_requires_release_and_rejects_old_packets(self):
        box=StageMailbox(2);a=Tag(0,0,0);x=box.grant(a)
        with self.assertRaises(ValueError):box.reset(0,1,state_cleared=True)
        self.ready(box,x,a);box.complete(x,a,send=True,credit=True)
        with self.assertRaises(ValueError):box.reset(0,1,state_cleared=False)
        box.reset(0,1,state_cleared=True)
        with self.assertRaises(ValueError):box.grant(a)
        y=box.grant(Tag(0,1,0));self.assertNotEqual(x,y)
        with self.assertRaises(ValueError):box.arrive(x,a,1)

    def test_autoregressive_feedback_and_context_bound(self):
        admission=FeedbackAdmission([[1,2],[3]],context=3)
        a=admission.admit(0,0,1);b=admission.admit(1,0,3)
        with self.assertRaises(ValueError):admission.admit(0,0,2)
        admission.selected(b,8,all_layers_and_head_completed=True)
        with self.assertRaises(ValueError):admission.admit(1,0,9)
        admission.admit(1,0,8)
        admission.selected(a,7,all_layers_and_head_completed=True)
        a=admission.admit(0,0,2)  # Still prefill: consumes real next prompt ID.
        with self.assertRaises(ValueError):admission.selected(a,9,all_layers_and_head_completed=False)
        admission.selected(a,9,all_layers_and_head_completed=True)
        a=admission.admit(0,0,9)
        admission.selected(a,10,all_layers_and_head_completed=True)
        with self.assertRaises(ValueError):admission.admit(0,0,10)
        with self.assertRaises(ValueError):admission.reset(0,1,range(65))
        admission.reset(0,1,range(66));admission.admit(0,1,1)
