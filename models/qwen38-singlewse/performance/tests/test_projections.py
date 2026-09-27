from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial.projections import bundle_graph, split_stream
from spatial.forest import pattern


def fixture():
    nodes = []
    for op, inputs, output, weight in [
        ('embedding_lookup', ['token'], 'x', 'embedding'),
        ('fp8_projection', ['x'], 'a', 'wa'), ('fp8_projection', ['x'], 'b', 'wb'),
        ('copy', ['a'], 'c', None), ('copy', ['b'], 'd', None), ('add', ['c', 'd'], 'z', None)]:
        nodes.append(dict(id=len(nodes), op=op, inputs=inputs, outputs=[output], weights=[weight] if weight else [], attributes={}))
    matrices = [dict(id=i, tensor=name, kind=kind, mode=mode, k_blocks=40, stream_start=start, tiles=40, shape=[2, 5120])
                for i, (name, kind, mode, start) in enumerate([
                    ('embedding', 'bf16', 'lookup', 0), ('wa', 'fp8', 'gemv', 0), ('wb', 'fp8', 'gemv', 40)])]
    values = {name: dict(base_cell=base, reserved_cells=32, birth=birth, last_use=end)
              for name, base, birth, end in [('x', 0, 0, 2), ('a', 32, 1, 3), ('b', 64, 2, 4),
                                           ('c', 96, 3, 5), ('d', 32, 4, 5), ('z', 128, 5, 6)]}
    return dict(model='test', revision='test', nodes=nodes), dict(matrices=matrices,
        classes={k: dict(pes=120, phase=0) for k in ('fp8', 'bf16')}, value_arena=dict(values=values))


class ProjectionTests(unittest.TestCase):
    def test_shared_input_fusion_preserves_rows_and_adds_storage_completion(self):
        graph, atlas = fixture(); p = bundle_graph(graph, atlas)
        self.assertEqual(p['events'][1]['original_nodes'], [1, 2])
        self.assertEqual(p['events'][1]['outputs'], ['a', 'b'])
        self.assertEqual(p['bundles'][1]['output_rows'], 4)
        # d only reads b but overwrites a: it must wait for c to finish reading a.
        self.assertEqual(p['events'][3]['data_dependencies'], [1])
        self.assertEqual(p['events'][3]['memory_dependencies'], [2])

    def test_different_preparation_or_noncontiguous_weights_do_not_fuse(self):
        for mutation in ('attributes', 'address'):
            graph, atlas = fixture()
            if mutation == 'attributes':
                graph['nodes'][2]['attributes']['different_rounding'] = True
            else:
                atlas['matrices'][2]['stream_start'] += 40
            p = bundle_graph(graph, atlas)
            self.assertEqual(p['events'][1]['original_nodes'], [1])
            self.assertEqual(p['events'][2]['original_nodes'], [2])

    def test_storage_class_identity_prevents_cross_class_fusion(self):
        graph,atlas=fixture();atlas['classes']['fp8-other']=dict(pes=120,phase=0)
        atlas['matrices'][2]['storage_class']='fp8-other'
        p=bundle_graph(graph,atlas)
        self.assertEqual(p['events'][1]['original_nodes'],[1])
        self.assertEqual(p['events'][2]['original_nodes'],[2])
        self.assertEqual(p['bundles'][2]['storage_class'],'fp8-other')

    def test_live_storage_alias_is_rejected(self):
        graph, atlas = fixture(); atlas['value_arena']['values']['d']['base_cell'] = 96
        with self.assertRaisesRegex(ValueError, 'live storage alias'):
            bundle_graph(graph, atlas)

    def test_resident_slot_wrap_is_explicit(self):
        segments = split_stream(80, 280, 120)
        self.assertEqual([(s['class_start'], s['count'], s['slot'], s['bundle_tile_start']) for s in segments],
                         [(80, 40, 0, 0), (0, 120, 1, 40), (0, 120, 2, 160)])

    def test_every_tree_color_has_balanced_endpoints_and_zero_boundary_cuts(self):
        for k in (40, 48, 136):
            p = pattern(k); vertices = p['vertices']; cuts = p['cuts']
            self.assertEqual((cuts[0], cuts[-1]), (0, 0))
            self.assertEqual(sum(v['source'].bit_count() for v in vertices), k - 1)
            for color in range(13):
                bit = 1 << color; live = False
                for i, v in enumerate(vertices):
                    self.assertEqual(bool(cuts[i] & bit), live)
                    if v['destination'] & bit:
                        self.assertFalse(live); live = True
                    self.assertEqual(bool(v['active'] & bit), live)
                    if v['source'] & bit:
                        self.assertTrue(live); live = False
                self.assertFalse(live)
