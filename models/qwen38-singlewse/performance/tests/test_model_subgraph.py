import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.model_subgraph import mlp_subgraph

class ModelSubgraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        r=Path(__file__).resolve().parents[2]
        cls.graph=json.loads((r/'configs/model-graph.json').read_text())
        cls.tensors=json.loads((r/'configs/tensors.json').read_text())['tensors']

    def test_real_full_dimensions_and_fork_lifetimes(self):
        d=mlp_subgraph(self.graph,self.tensors)
        self.assertEqual([n['id'] for n in d['original_nodes']],[13,14,15,16,17,18])
        self.assertEqual([p['shape'] for p in d['projections']],[[17408,5120],[17408,5120],[5120,17408]])
        self.assertEqual(d['resource_counts']['original_weight_bytes'],267429760)
        self.assertEqual(d['buffer_lifetimes']['v16']['consumers'],[13,18])
        self.assertEqual(d['buffer_lifetimes']['quantized:v17']['consumers'],[14,15])
        self.assertEqual(d['buffer_lifetimes']['v22']['consumers'],['interface:output:v22'])
        self.assertEqual(d['completion_dependencies']['13'],['interface:input:v16'])
        self.assertEqual(d['real_successor_interfaces'][0]['node'],19)
        self.assertFalse(d['executable'])

    def test_changed_fork_semantics_rejected_and_original_not_mutated(self):
        before=copy.deepcopy(self.graph);mlp_subgraph(self.graph,self.tensors);self.assertEqual(before,self.graph)
        bad=copy.deepcopy(self.graph);bad['nodes'][15]['inputs']=['v18']
        with self.assertRaises(AssertionError):mlp_subgraph(bad,self.tensors)
