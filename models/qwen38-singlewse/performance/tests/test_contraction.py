from dataclasses import replace
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.contraction import ContractionPlan,tree
class ContractionTests(unittest.TestCase):
 def test_full_width_partition_and_express_endpoints(self):
  for mesh in (0,4,8):
   p=ContractionPlan(mesh_width=mesh);r=p.validate();self.assertEqual(r['active_pes'],224)
   routes={(v['x'],v['y'],v['color']):v for v in p.routes()}
   steps={'EAST':(1,0),'WEST':(-1,0),'NORTH':(0,-1),'SOUTH':(0,1)};all_owners=[]
   for group,n in enumerate(p.blocks):
    nodes=tree(n);owners=[p.coordinate(group,k) for k in range(n)];all_owners+=owners
    self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(owners,owners[1:])))
    def visit(rank):
     out=[rank]
     for child in nodes[rank]['children']:out+=visit(child)
     self.assertEqual(out,list(range(rank,rank+nodes[rank]['size'])));return out
    self.assertEqual(visit(0),list(range(n)))
    for v in nodes[1:]:
     xy=owners[v['rank']];c=p.color(v);self.assertEqual(routes[*xy,c]['rx'],'RAMP');seen=set()
     while routes[*xy,c]['tx']!=['RAMP']:
      self.assertNotIn(xy,seen);seen.add(xy);dx,dy=steps[routes[*xy,c]['tx'][0]];xy=(xy[0]+dx,xy[1]+dy)
     self.assertEqual(xy,owners[v['parent']])
    for k in range(1,n):
     c=p.chain_colors[k%2];self.assertEqual(routes[*owners[k],c]['rx'],'RAMP');self.assertEqual(routes[*owners[k-1],c]['tx'],['RAMP'])
   self.assertEqual(len(set(all_owners)),224)
   if mesh:self.assertEqual(set(all_owners),{(x,y) for x in range(p.width) for y in range(p.height)})
 def test_reject_resources_and_partial_width(self):
  for p in [replace(ContractionPlan(),blocks=(8,)),replace(ContractionPlan(),chain_colors=(3,18)),replace(ContractionPlan(),dsrs=(3,4,6)),replace(ContractionPlan(),input_queues=(2,3,3)),replace(ContractionPlan(),mesh_width=7)]:
   with self.assertRaises(ValueError):p.validate()
