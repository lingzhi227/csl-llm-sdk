import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.model_subgraph import mlp_subgraph
from spatial.mlp_regions import MlpRegions,MlpRegionSpec
from spatial.region_epoch import RegionEpoch,GroupReadiness
from tools.audit_mlp_regions import audit


class MlpRegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r=Path(__file__).resolve().parents[2]
        cls.semantic=mlp_subgraph(json.loads((r/'configs/model-graph.json').read_text()),
                                  json.loads((r/'configs/tensors.json').read_text())['tensors'])
        cls.flow=MlpRegions(cls.semantic)

    def test_full_original_tensor_coverage_and_known_route_admission(self):
        d=self.flow.document();a=audit(d,self.flow.original_tiles(),self.flow.native_edges())
        self.assertEqual(d['application'],[747,1080])
        self.assertEqual(a['original_matrix_tiles'],1044480)
        self.assertEqual(a['distinct_active_pes'],711224)
        self.assertEqual(a['known_directed_link_max_words'],128)
        self.assertEqual(d['memory']['prior_atlas_comparison']['deficit_if_all_helpers_exclude_weight_banks'],5784)
        self.assertFalse(a['all_streams_audited']);self.assertFalse(d['executable'])

    def test_geometry_budget_rejects_before_lowering(self):
        self.assertEqual(MlpRegions(self.semantic,MlpRegionSpec(columns=8)).height,1145)
        for spec in [MlpRegionSpec(columns=7),MlpRegionSpec(columns=10),MlpRegionSpec(native_rows=8),
                     MlpRegionSpec(down_chunk_words=3),MlpRegionSpec(maximum_return_stream_words=127)]:
            with self.assertRaises(ValueError):MlpRegions(self.semantic,spec)

    def test_tile_address_and_coordinate_faults_rejected(self):
        for fault in ['address','coordinate']:
            def bad():
                for i,t in enumerate(self.flow.original_tiles()):
                    if i==0:
                        if fault=='address':t['rows']=[2,4]
                        else:t['xy']=(t['xy'][0]+1,t['xy'][1])
                    yield t
            with self.assertRaises(AssertionError):audit(self.flow.document(),bad(),self.flow.native_edges())

    def test_native_parent_and_header_fork_faults_rejected(self):
        def bad_native():
            for i,s in enumerate(self.flow.native_edges()):
                if i==0:s['destination'][0]-=1
                yield s
        with self.assertRaises(AssertionError):audit(self.flow.document(),self.flow.original_tiles(),bad_native())
        d=self.flow.document();d['input_distribution'][0]['consumers'].pop()
        with self.assertRaises(AssertionError):audit(d,self.flow.original_tiles(),self.flow.native_edges())

    def test_fusion_keeps_quantization_rounding_and_real_successor(self):
        d=self.flow.document()['fusion'];decisions={f['id']:f for f in d['decisions']}
        f=decisions['paired-projection-activation']
        self.assertEqual(f['nodes'],[14,15,16])
        self.assertEqual(f['no_global_materialization'],['v18','v19'])
        self.assertIn('SiLU BF16 before multiply',f['preserves'])
        self.assertEqual(decisions['activation-quant-down']['items_per_owner'],128)
        self.assertEqual(decisions['down-residual-successor']['actual_successors'][0]['node'],19)
        self.assertFalse(d['complete_fused_runtime'])

    def test_protocol_waits_for_forwarding_drain_and_queue_reset(self):
        for native_first in [False,True]:
            e=RegionEpoch();e.begin(0)
            (e.sent_native if native_first else e.delivered_row)()
            with self.assertRaises(ValueError):e.routes_installed()
            (e.delivered_row if native_first else e.sent_native)()
            e.routes_installed()
            with self.assertRaises(ValueError):e.acknowledge()
            with self.assertRaises(ValueError):e.bind_first(9)
            e.bind_first(4);e.acknowledge();e.send_chunk(0)
            with self.assertRaises(ValueError):e.send_chunk(0)
            e.consumed()
            with self.assertRaises(ValueError):e.release()
            e.sent_chunk();e.release();e.begin(1)
            self.assertIsNone(e.first_color)
            with self.assertRaises(ValueError):e.send_chunk(0)

    def test_warm_epochs_both_consumer_callback_orders(self):
        e=RegionEpoch()
        for epoch in range(3):
            e.begin(epoch);e.delivered_row();e.sent_native();e.routes_installed();e.bind_first(4);e.acknowledge()
            for i in range(16):
                e.send_chunk(epoch)
                if i==15 and epoch%2:e.consumed()
                e.sent_chunk()
            if not epoch%2:e.consumed()
            e.release()
        with self.assertRaises(ValueError):e.begin(4)

    def test_early_group_packet_and_duplicate_readiness_rejected(self):
        g=GroupReadiness()
        for row in range(64):g.arrive(row,value=True,ready=row!=17)
        self.assertFalse(g.may_quantize())
        with self.assertRaises(ValueError):g.arrive(18,ready=True)
        g.arrive(17,ready=True);self.assertTrue(g.may_quantize())
