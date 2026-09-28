import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.layer_routes import audit_mlp_network,contraction_groups,reduction_tree,worker,fused_deliveries
from spatial.layer_schedule import tile_owner


class LayerRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan=json.loads((Path(__file__).resolve().parents[1]/'evidence/layer-native-schedule-002/layer-schedule.json').read_text())

    def test_all_original_mlp_reductions_and_fusion_endpoints(self):
        a=audit_mlp_network(self.plan)
        self.assertTrue(a['passed']);self.assertEqual(a['regions'],128)
        self.assertEqual(a['workers'],502320);self.assertEqual(a['edges'],499568)
        self.assertEqual(a['fused_projection_blocks'],64*17408//8)
        self.assertEqual(a['tree_colors'],list(range(3,18)))
        self.assertFalse(a['all_layer_routes_admitted'])
        self.assertFalse(a['root_consumer_routes_connected'])

    def test_ragged_trees_have_each_original_k_slice_once(self):
        for workers,blocks in [(106,160),(144,160),(159,272),(160,160),(184,272),(272,272)]:
            root,nodes=reduction_tree(workers,blocks)
            self.assertEqual(root,workers//2)
            def reduce_node(rank):
                n=nodes[rank];parts=set(range(rank,blocks,workers))
                for child in n['children']:
                    more=reduce_node(child);self.assertFalse(parts&more);parts|=more
                self.assertEqual(len(parts),n['subtree_k']);return parts
            self.assertEqual(reduce_node(root),set(range(blocks)))

    def test_runtime_bases_cover_full_rows_and_original_auxiliary_bytes(self):
        for r in self.plan['stages'][1]['regions'][1:]:
            for g in contraction_groups(r):
                for n in g['nodes']:
                    w=worker(r,g,n);arm=w['arm'];rank=w['rank']
                    c=next(c for c in r['banks']['classes'] if c['rank_start']<=rank<c['rank_end'])
                    prefix=c['page_prefix']+(rank-c['rank_start'])*c['auxiliary_pages']
                    pages=set(range(prefix,min(prefix+c['auxiliary_pages'],r['auxiliary_pages'])))
                    expected=c['fp8_slots']*260+c['bf16_slots']*256+128*len(pages)
                    self.assertEqual(w['parameters']['bank_words']*4,expected)
                    for branch,m in enumerate(r['matrices']):
                        for iteration in [0,g['output_count']-1]:
                            for part,k in enumerate(w['native_input_slices']):
                                owner=tile_owner(r,branch,(g['output_start']+iteration)*m['k_blocks']+k)
                                base=arm['base0'] if branch==0 else arm['base1']
                                self.assertEqual(owner['byte_offset'],4*(base+(iteration*n['parts']+part)*65))
                                self.assertLessEqual(owner['byte_offset']+260,expected)

    def test_cross_root_quantization_groups_have_contiguous_pairs(self):
        r=self.plan['stages'][1]['regions'][1];by_group={}
        for d in fused_deliveries(r):by_group.setdefault(d['quantization_group'],[]).append(d)
        crossed=0
        for group,records in by_group.items():
            indices=[d['first_pair']+i for d in records for i in range(d['pairs'])]
            self.assertEqual(sorted(indices),list(range(64)))
            self.assertEqual(len({tuple(d['destination']) for d in records}),1)
            if len({d['producer'] for d in records})>1:crossed+=1
            self.assertTrue(all(d['physical_path'] is None for d in records))
        self.assertGreater(crossed,0)

    def test_missing_original_rows_are_rejected(self):
        bad=copy.deepcopy(self.plan);r=bad['stages'][1]['regions'][1]
        for m in r['matrices']:m['group_partitions'][0]['output_count']-=1
        with self.assertRaises(ValueError):audit_mlp_network(bad)


if __name__=='__main__':unittest.main()
