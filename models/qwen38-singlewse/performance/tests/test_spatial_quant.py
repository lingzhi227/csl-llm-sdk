from dataclasses import replace
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.quant import SpatialQuantPlan

class SpatialQuantTests(unittest.TestCase):
 def test_tree_routes_and_order(self):
  steps={'EAST':(1,0),'WEST':(-1,0),'SOUTH':(0,1),'NORTH':(0,-1)}
  for w,h in [(2,2),(2,4),(4,4),(8,4),(4,8),(8,8),(16,4),(4,16)]:
   p=SpatialQuantPlan(width=w,height=h);p.validate();routes={(r['x'],r['y'],r['color']):r for r in p.routes()}
   # Traverse actual emitted scale routes, independently of subtree helper.
   pending=[(0,0)];seen=set();receivers=set()
   while pending:
    xy=pending.pop();self.assertNotIn(xy,seen);seen.add(xy)
    for d in routes[*xy,p.scale_color]['tx']:
     if d=='RAMP':receivers.add(xy)
     else:
      dx,dy=steps[d];pending.append((xy[0]+dx,xy[1]+dy))
   self.assertEqual(receivers,seen-{(0,0)});self.assertEqual(len(seen),w*h)
   def traversal(x,y):
    out=[y*w+x]
    for cx,cy in p.children(x,y):out+=traversal(cx,cy)
    return out
   self.assertEqual(traversal(0,0),list(range(w*h)))
   for x,y in seen:
    self.assertEqual(traversal(x,y),p.subtree(x,y))
    if x or y:
     parent=(x-1,y) if x else (0,y-1)
     for rc,cc in [(p.max_row_colors,p.max_column_colors),(p.gather_row_colors,p.gather_column_colors)]:
      color=rc[x%2] if x else cc[y%2]
      r=routes[x,y,color];self.assertEqual(len(r['tx']),1)
      dx,dy=steps[r['tx'][0]];self.assertEqual((x+dx,y+dy),parent)
   self.assertEqual(p.items*w*h,128);self.assertEqual(p.items%2,0)
 def test_reject_conflicting_resources(self):
  for p in [replace(SpatialQuantPlan(),width=3),replace(SpatialQuantPlan(),scale_color=3),replace(SpatialQuantPlan(),input_queues=(2,3,4,5,5)),replace(SpatialQuantPlan(),dsrs=(2,4,5,6,7))]:
   with self.assertRaises(ValueError):p.validate()
