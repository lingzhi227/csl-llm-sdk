import copy
from dataclasses import replace
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.projection_flow import ProjectionFlow,ProjectionSpec
from tools.audit_projection_flow import audit
from spatial.sdk_leases import audit_explicit_sdk_leases

class ProjectionFlowTests(unittest.TestCase):
    def test_both_physical_axis_mappings_pass_independent_traversal(self):
        for axis,shape in [('x',[41,25]),('y',[25,41])]:
            d=ProjectionFlow(ProjectionSpec(k_axis=axis)).document();self.assertEqual(d['application'],shape)
            a=audit(d);self.assertEqual(a['input_owners'],40);self.assertEqual(a['workers'],960)
            self.assertEqual(a['max_link_words_including_teardown'],66);self.assertEqual(a['max_input_hops'],24)

    def test_cost_filters_and_sdk_colors_reject_before_compile(self):
        for s in [replace(ProjectionSpec(),max_input_hops=20),replace(ProjectionSpec(),max_link_words=65),replace(ProjectionSpec(),max_contribution_transport_hops=127),replace(ProjectionSpec(),tile_rows=8,tile_columns=32)]:
            with self.assertRaises(ValueError):ProjectionFlow(s).document()

    def test_route_traffic_and_completion_faults_are_rejected(self):
        d=ProjectionFlow().document()
        for mode in ['traffic','route','dependency','partition']:
            bad=copy.deepcopy(d)
            if mode=='traffic':bad['cost']['link_loads'][0]['words']+=1
            if mode=='route':bad['routes'][1]['rx']='NORTH'
            if mode=='dependency':bad['completion_dag']['sum:0']=['input:0']
            if mode=='partition':bad['workers'][3]['column_start']+=1
            with self.assertRaises((AssertionError,KeyError)):audit(bad)

    def test_shared_protocol_explicit_resources_avoid_sdk(self):
        r=Path(__file__).resolve().parents[1]
        sources={'matrix_epoch.csl':(r/'protocols/matrix_epoch.csl').read_bytes()}
        for name in ['fp8_shape.csl','bf16_shape.csl']:sources[name]=(r/'probes/native_shapes'/name).read_bytes()
        self.assertTrue(audit_explicit_sdk_leases(sources)['passed'])
