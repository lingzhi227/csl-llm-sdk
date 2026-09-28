"""Actual full projection packets and adversarial routing mutations."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial.mixer_frontend_network import lower_frontend_network, audit_frontend_network, frontend_index


class FrontendNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=ROOT/'evidence/layer-mlp-compile-021/source'
        cls.region=next(r for r in json.loads((cls.source/'joint-stage.json').read_text())['regions'] if r['role']=='mix')
        cls.profiles=[dict(pe=pe,source=p['source'],parameters=dict(p['parameters']))
                      for p in json.loads((cls.source/'profiles.json').read_text())['profiles'] for pe in p['pes']]
        cls.setups=json.loads((cls.source/'mixer-setups.json').read_text())
        cls.plan=lower_frontend_network(cls.region,cls.profiles,cls.setups)

    def test_all_original_values_reach_one_correct_group(self):
        audit=audit_frontend_network(self.plan,self.setups)
        self.assertEqual((audit['original_projection_values'],audit['projected_packets'],audit['root_to_consumer_paths']),
                         (16480,8288,1408))
        self.assertFalse(audit['physical'])
        for matrix,size in enumerate([10240,6144,48,48]):
            for row in range(0,size,2 if matrix<2 else 1):
                group,index=frontend_index(matrix,row)
                # Independent semantic head decoding, including every boundary.
                if matrix==0:
                    head=row//128
                    want=head if head<16 else head-16 if head<32 else (head-32)//3
                elif matrix==1:want=(row//128)//3
                else:want=row//3
                self.assertEqual(group,want)
                self.assertEqual(index,[0,10240,16384,16432][matrix]+row)
        self.assertTrue(all((h*43)>>7 == h//3 for h in range(48)))

    def test_no_existing_fabric_route_or_norm_queue_is_claimed(self):
        # Parse the exact qualified emitted route configurations, independently
        # of the new planner. All old and new coordinates are stage-global.
        layout=(self.source/'layout.csl').read_text()
        arrays={name:list(map(int,values.split(','))) for name,values in
                re.findall(r'const (route_\d+)=\[\d+\]u16\{([0-9,]+)\}',layout)}
        existing=set()
        for name,stride,color in re.findall(r'@set_color_config\((route_\d+)\[(\d+)\*i\],route_\d+\[\d+\*i\+1\],@get_color\((\d+)\)',layout):
            v=arrays[name];stride=int(stride)
            existing.update((v[i]+63,v[i+1],int(color)) for i in range(0,len(v),stride))
        self.assertGreater(len(existing),60000)
        new={(*r['pe'],r['color']) for r in self.plan['routes']}
        self.assertFalse(existing & new)
        self.assertTrue(all(m['pe'][1]<143 for m in self.plan['mergers']))
        self.assertEqual({tuple(c['pe']) for c in self.plan['consumers']},{(x,145) for x in range(74,90)})

    def test_missing_duplicate_wrong_filter_and_merge_are_rejected(self):
        for change in ('missing','duplicate','multi_rx','filter','color','consumer','merge','producer','source_descriptor'):
            plan=deepcopy(self.plan);setups=deepcopy(self.setups)
            if change=='missing':plan['routes'].pop(0)
            if change=='duplicate':plan['routes'].append(deepcopy(plan['routes'][0]))
            if change=='multi_rx':plan['routes'][0]['rx'].append('NORTH')
            if change=='filter':next(r for r in plan['routes'] if 'filter_tag' in r)['filter_tag']=16
            if change=='color':plan['routes'][0]['color']=10
            if change=='consumer':plan['consumers'][0]['group']=1
            if change=='merge':plan['mergers'][0]['outgoing']=plan['mergers'][0]['incoming']
            if change=='producer':plan['producers'][0]['color']=19
            if change=='source_descriptor':setups[0][1][2]+=2
            with self.subTest(change=change),self.assertRaises(ValueError):audit_frontend_network(plan,setups)

    def test_semantic_input_cannot_alias_a_head_or_projection(self):
        for args in [(0,1),(0,10240),(1,6144),(2,48),(3,-1),(4,0),(True,0),(0,False)]:
            with self.assertRaises(ValueError):frontend_index(*args)
        profiles=deepcopy(self.profiles)
        next(p for p in profiles if p['pe']==[74,145])['source']='device_mixer_layer_norm_bridge.csl'
        with self.assertRaises(ValueError):lower_frontend_network(self.region,profiles,self.setups)


if __name__=='__main__':unittest.main()
