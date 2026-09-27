from dataclasses import replace
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.mixed_bank import MixedBankPlan
from spatial.contraction import tree

class MixedBankTests(unittest.TestCase):
 def test_every_tree_packet_reaches_its_only_parent(self):
  for count in (2,6,8,16):
   p=MixedBankPlan(blocks=(count,));p.validate();nodes=tree(count);routes={(r['x'],r['y'],r['color']):r for r in p.routes()};steps={'EAST':(1,0),'WEST':(-1,0),'NORTH':(0,-1),'SOUTH':(0,1)}
   for node in nodes[1:]:
    xy=p.coordinate(0,node['rank']);color=p.color(node);seen=set();self.assertEqual(routes[*xy,color]['rx'],'RAMP')
    while routes[*xy,color]['tx']!=['RAMP']:
     self.assertNotIn(xy,seen);seen.add(xy);dx,dy=steps[routes[*xy,color]['tx'][0]];xy=(xy[0]+dx,xy[1]+dy)
    self.assertEqual(xy,p.coordinate(0,node['parent']))
   self.assertFalse(any(r['color'] in p.chain_colors for r in p.routes()))
  self.assertEqual(sorted(set(len(v['children']) for v in tree(6))),[0,1,2])
 def test_reject_descriptor_collision_and_capacity_reduction(self):
  for p in [replace(MixedBankPlan(),dsrs=(3,5,7)),replace(MixedBankPlan(),dsrs=(3,4,6)),replace(MixedBankPlan(),fp8_slots=111),replace(MixedBankPlan(),bf16_slots=11),replace(MixedBankPlan(),blocks=(5,)),replace(MixedBankPlan(),tasks=(8,8,10))]:
   with self.assertRaises(ValueError):p.validate()
