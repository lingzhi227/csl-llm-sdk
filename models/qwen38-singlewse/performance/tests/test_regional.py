from pathlib import Path
from dataclasses import replace
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.regional import RegionalPlan

class RegionalTests(unittest.TestCase):
 def endpoints(self,p,origin,color):
  routes={(r['x'],r['y'],r['color']):r for r in p.routes()}
  pending=[origin];seen=set();ends=[];deltas={'EAST':(1,0),'WEST':(-1,0),'SOUTH':(0,1),'NORTH':(0,-1)}
  while pending:
   x,y=pending.pop();self.assertNotIn((x,y),seen);seen.add((x,y))
   for d in routes[x,y,color]['tx']:
    if d=='RAMP':ends.append((x,y))
    else:
     dx,dy=deltas[d];pending.append((x+dx,y+dy))
  return set(ends)
 def test_multicast_and_result_return_endpoints(self):
  for w,h in [(2,2),(8,8),(40,3)]:
   p=RegionalPlan(width=w,height=h);p.validate()
   for x in range(w):self.assertEqual(self.endpoints(p,(x,0),p.operand_color),{(x,y) for y in range(1,h)})
   for y in range(h):self.assertEqual(self.endpoints(p,(w-1,y),p.return_color),{(0,y)})
 def test_each_sum_and_ack_is_one_owned_neighbor(self):
  p=RegionalPlan(width=8,height=8)
  for y in range(p.height):
   for x in range(p.width-1):self.assertEqual(self.endpoints(p,(x,y),p.sum_colors[x%2]),{(x+1,y)})
  for y in range(1,p.height):self.assertEqual(self.endpoints(p,(0,y),p.ack_colors[y%2]),{(0,y-1)})
 def test_resource_and_geometry_conflicts_rejected(self):
  for p in [replace(RegionalPlan(),input_queues=(2,3,3,5)),replace(RegionalPlan(),dsrs=(0,4,5,6,7)),replace(RegionalPlan(),return_color=3),replace(RegionalPlan(),width=1)]:
   with self.assertRaises(ValueError):p.validate()
