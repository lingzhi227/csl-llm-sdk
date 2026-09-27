from dataclasses import replace
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.epoch_bank import EpochBankPlan

class EpochBankTests(unittest.TestCase):
 def test_multicast_all_workers_and_return_only_controller(self):
  p=EpochBankPlan();p.validate();routes={(r['x'],r['y'],r['color']):r for r in p.routes()};steps={'NORTH':(0,-1),'SOUTH':(0,1),'WEST':(-1,0),'EAST':(1,0)}
  def endpoints(origin,color):
   pending=[origin];visited=set();ends=[]
   while pending:
    xy=pending.pop();self.assertNotIn(xy,visited);visited.add(xy)
    for direction in routes[*xy,color]['tx']:
     if direction=='RAMP':ends.append(xy)
     else:dx,dy=steps[direction];pending.append((xy[0]+dx,xy[1]+dy))
   return ends
  self.assertEqual(set(endpoints((0,3),p.request_color)),{p.coordinate(0,k) for k in range(6)})
  self.assertEqual(endpoints((0,0),p.result_color),[(0,3)])
  self.assertEqual(p.validate()['application_pes'],8)
 def test_reject_event_lease_collisions(self):
  for p in [replace(EpochBankPlan(),worker_uts=(2,3,4,4)),replace(EpochBankPlan(),controller_uts=(2,2)),replace(EpochBankPlan(),request_dsr=3),replace(EpochBankPlan(),request_iq=2),replace(EpochBankPlan(),request_task=9),replace(EpochBankPlan(),request_color=3),replace(EpochBankPlan(),result_color=18),replace(EpochBankPlan(),fp8_slots=111)]:
   with self.assertRaises(ValueError):p.validate()
