"""Walk emitted router records independently of the planned path lists."""
from collections import Counter
from copy import deepcopy
import json
import unittest

from tools.build_resident_gdn_stage import build, retained_routes
from spatial.resident_gdn_routes import lower

DELTA = {'EAST':(1,0),'WEST':(-1,0),'NORTH':(0,-1),'SOUTH':(0,1)}
OPPOSITE = {'EAST':'WEST','WEST':'EAST','NORTH':'SOUTH','SOUTH':'NORTH'}


def neighbor(pe, direction):
    dx,dy = DELTA[direction]
    return pe[0]+dx,pe[1]+dy


class ResidentRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files,_,_,cls.profiles = build((0,1,2))
        cls.plan = json.loads(cls.files['resident-gdn-routes.json'])
        cls.placement = json.loads(cls.files['gdn-bank-placement.json'])
        cls.workers = {tuple(w['pe']):w for w in cls.placement['workers']}

    def test_original_routes_are_not_displaced(self):
        original = retained_routes(self.files)
        all_routes = original+self.plan['routes']
        self.assertEqual(len(all_routes),len({(*r['pe'],r['color']) for r in all_routes}))
        self.assertEqual(len(original),75317)
        self.assertEqual(self.plan['new_software_relays'],0)
        self.assertFalse(self.plan['all_groups_routed'])

    def test_counter_broadcast_and_switch_returns(self):
        installed = {(*r['pe'],r['color']):r for r in self.plan['routes']}
        for chain in self.plan['chains']:
            source = tuple(chain['frontend']); pe = source; incoming = 'RAMP'
            forward_color = chain['input_color']; return_color = chain['return_color']
            visited = set(); worker_order = []
            while True:
                self.assertNotIn(pe,visited);visited.add(pe)
                r = installed[(*pe,forward_color)]
                self.assertEqual(r['rx'],[incoming])
                if 'RAMP' in r['tx']:
                    worker_order.append(pe);w=self.workers[pe]
                    self.assertEqual(w['head']//3,chain['group'])
                    c = r['counter'];count=c['initial'];accepted=[]
                    for token in range(2):
                        for head in range(3):
                            for word in range(387):
                                if count<=c['maximum']:accepted.append((token,head,word))
                                count=0 if count==c['limit'] else count+1
                    self.assertEqual(accepted,[(t,w['head']%3,k)for t in range(2)for k in range(387)])
                onward = [d for d in r['tx'] if d!='RAMP']
                if not onward:break
                self.assertEqual(len(onward),1)
                d=onward[0];pe=neighbor(pe,d);incoming=OPPOSITE[d]
            expected = {pe for pe,w in self.workers.items() if w['head']//3==chain['group']}
            self.assertEqual(set(worker_order),expected)
            # All workers can be ready in any order. The static return switches
            # allow exactly this order; each local ADV opens its child's path.
            advanced=set();rows=Counter();markers=0
            for origin in worker_order:
                w=self.workers[origin]
                for first in range(w['first'],w['first']+w['columns'],2):
                    rows.update([(w['head'],first),(w['head'],first+1)])
                pe=origin;arrival='RAMP';seen=set()
                while True:
                    self.assertNotIn(pe,seen);seen.add(pe)
                    r=installed[(*pe,return_color)]
                    selected = [r['switch']['next_rx']] if pe in advanced else r['rx']
                    self.assertEqual(selected,[arrival])
                    self.assertEqual(len(r['tx']),1)
                    d=r['tx'][0]
                    if d=='RAMP':
                        self.assertEqual(pe,source);break
                    pe=neighbor(pe,d);arrival=OPPOSITE[d]
                # Pop only the originating ADV; ancestor switches stay open.
                self.assertTrue(installed[(*origin,return_color)]['switch']['pop_on_advance'])
                advanced.add(origin);markers+=1
            self.assertEqual(markers,chain['return_markers'])
            self.assertEqual(rows,Counter({(h,v):1 for h in range(3*chain['group'],3*chain['group']+3)for v in range(128)}))

    def test_unsupported_groups_and_endpoint_collisions_fail_closed(self):
        frontends={p['parameters']['frontend_group']:p['pe']for p in self.profiles if 'frontend_group'in p['parameters']}
        args=(self.placement['stage'],self.placement['workers'],frontends)
        with self.assertRaises(ValueError):lower(*args,retained_routes(self.files),(3,))
        collision=deepcopy(retained_routes(self.files))
        collision.append(dict(pe=frontends[0],color=14,rx=['WEST'],tx=['EAST']))
        with self.assertRaises(ValueError):lower(*args,collision,(0,))


if __name__=='__main__':
    unittest.main()
