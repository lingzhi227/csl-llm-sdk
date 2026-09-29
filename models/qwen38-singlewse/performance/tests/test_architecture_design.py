import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_schedule import audit_native_schedule,xy_rank
from prediction.placement_paths import manhattan,owner_flows
from spatial.pipeline import audit_pipeline
from spatial.pipeline_lowering import lower_pipeline,audit_boundaries


class ArchitectureDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'performance/evidence/dialogue-architecture-001'
        cls.plan=json.loads((base/'selected-plan.json').read_text())
        cls.selection=json.loads((base/'selection.json').read_text())
        cls.tensors=json.loads((ROOT/'configs/tensors.json').read_text())['tensors']

    def test_complete_original_native_banks_and_extended_state(self):
        result=audit_native_schedule(self.plan,self.tensors,require_controller=False)
        self.assertTrue(result['passed']);self.assertFalse(result['controller_placement_checked'])
        self.assertEqual((result['regions'],result['matrices'],result['native_tiles']),(194,498,115077120))
        self.assertEqual(self.plan['state_bytes_per_request'],48*48*128*128*4+48*3*10240*2+16*512*4096)
        for stage in self.plan['stages']:
            for region in stage['regions']:
                cursor=0
                for a in region['auxiliary']:
                    self.assertEqual(a['page_start'],cursor);cursor+=a['pages']
                    self.assertEqual(a['pages'],(a['bytes']+127)//128)
                self.assertEqual(cursor,region['auxiliary_pages'])
                self.assertLessEqual(cursor,region['banks']['auxiliary_page_capacity'])

    def test_controller_budget_has_actual_coordinate_and_excludes_fusion_actor(self):
        self.assertEqual(len(self.selection['controllers']),66)
        for stage,controller in zip(self.plan['stages'],self.selection['controllers']):
            region=next(r for r in stage['regions']if r['id']==controller['region'])
            self.assertEqual(xy_rank(region,controller['pe']),controller['rank'])
            self.assertNotIn(controller['rank'],{a['rank']for a in region.get('quantization_actors',[])})
            self.assertGreaterEqual(controller['minimum_planned_margin_bytes'],0)
        self.assertFalse(self.selection['controller_runtime_qualified'])
        with self.assertRaises(KeyError):audit_native_schedule(self.plan,self.tensors)

    def test_protocol_context_matches_selected_state_capacity(self):
        graph=json.loads((ROOT/'configs/model-graph.json').read_text())
        ir=lower_pipeline(self.plan,graph,self.tensors)
        self.assertEqual(ir['request_isolation']['context'],512)
        self.assertTrue(all(k['protocol_params']['context']==512 for k in ir['kernels']))

    def test_native_bank_corruption_is_rejected(self):
        damaged=copy.deepcopy(self.plan)
        damaged['stages'][1]['regions'][1]['banks']['classes'][0]['fp8_slots']+=1
        with self.assertRaises(ValueError):audit_native_schedule(damaged,self.tensors,require_controller=False)

    def test_all_handoff_rows_are_owned_once_and_fit_original_slot_page(self):
        rows={}
        for f in owner_flows(self.plan):
            key=f['source_stage'];seen=rows.setdefault(key,set())
            self.assertFalse(seen.intersection(range(f['first'],f['first']+f['count'])))
            seen.update(range(f['first'],f['first']+f['count']))
            self.assertLessEqual(f['count'],64)
            self.assertEqual(f['first']//64,(f['first']+f['count']-1)//64)
        self.assertEqual(len(rows),63)
        self.assertTrue(all(v==set(range(5120))for v in rows.values()))

    def test_paths_include_actual_neighbor_hops(self):
        for a,b in [((0,0),(3,4)),((3,4),(0,0)),((5,5),(5,5))]:
            path=manhattan(a,b);self.assertEqual((path[0],path[-1]),(a,b))
            self.assertEqual(len(path)-1,sum(abs(x-y)for x,y in zip(a,b)))
            self.assertTrue(all(sum(abs(x-y)for x,y in zip(s,t))==1 for s,t in zip(path,path[1:])))

    def test_legal_nonadjacent_region_stream_remains_explicitly_unqualified(self):
        path=ROOT/'performance/evidence/architecture-screen-010/stage-map.json'
        plan=json.loads(path.read_text());graph=json.loads((ROOT/'configs/model-graph.json').read_text())
        self.assertTrue(audit_pipeline(plan,graph,self.tensors)['admitted'])
        ir=lower_pipeline(plan,graph,self.tensors);a=audit_boundaries(ir)
        self.assertEqual(a['nonadjacent_internal_streams'],3)
        self.assertFalse(a['complete_fabric_routes_admitted'])
        self.assertTrue(all(not e['boundary_ports']for e in ir['internal_streams']if e['nonadjacent_route_required']))


if __name__=='__main__':unittest.main()
