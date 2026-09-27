import copy
from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.model import import_graph,load_original
ROOT=Path(__file__).resolve().parents[2]

class ModelTests(unittest.TestCase):
    def test_all_original_operators_and_state_edges(self):
        r=load_original(ROOT)
        self.assertEqual(r['metrics']['original_tensors'],1251)
        self.assertEqual(r['metrics']['original_weight_bytes'],29468003328)
        self.assertEqual(len(r['regions']),66)
        for node in r['nodes']:
            self.assertTrue(all(d<node['id'] for d in node['dependencies']))
            if node['op']=='full_causal_attention':
                deps=[r['nodes'][d]['op'] for d in node['dependencies']]
                self.assertIn('kv_append',deps)
        delayed=[e for e in r['edges'] if e['delay_tokens']==1]
        self.assertEqual(len(delayed),48+48+16+1)
        self.assertFalse(r['executable_full_model'])
    def test_missing_layer_and_wrong_weight_revision_rejected(self):
        graph=json.loads((ROOT/'configs/model-graph.json').read_text())
        tensors=json.loads((ROOT/'configs/tensors.json').read_text())
        bad=copy.deepcopy(graph);bad['revision']='wrong'
        with self.assertRaises(ValueError):import_graph(bad,tensors)
        bad=copy.deepcopy(graph);bad['nodes'].pop(2)
        with self.assertRaises(ValueError):import_graph(bad,tensors)
    def test_undefined_tensor_rejected(self):
        graph=json.loads((ROOT/'configs/model-graph.json').read_text())
        tensors=json.loads((ROOT/'configs/tensors.json').read_text())
        graph['nodes'][2]['inputs']=['undefined']
        with self.assertRaises(ValueError):import_graph(graph,tensors)
