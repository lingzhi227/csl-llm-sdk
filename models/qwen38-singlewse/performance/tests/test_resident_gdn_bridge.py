"""Independent route/word/retirement coverage of the original bridged cohorts."""
from collections import Counter
import json,unittest
from tools.build_resident_bridge_stage import build
from tools.build_resident_gdn_stage import retained_routes
from tools.build_frontend_stage import verified_frozen
from pathlib import Path
DELTA={'EAST':(1,0),'WEST':(-1,0),'NORTH':(0,-1),'SOUTH':(0,1)}
OPPOSITE={'EAST':'WEST','WEST':'EAST','NORTH':'SOUTH','SOUTH':'NORTH'}
def neighbor(pe,d):return tuple(a+b for a,b in zip(pe,DELTA[d]))

class BridgeRoutes(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.files,_,_,cls.profiles=build();cls.plan=json.loads(cls.files['resident-gdn-bridge-routes.json'])
  cls.workers={tuple(w['pe']):w for w in json.loads(cls.files['gdn-bank-placement.json'])['workers']}
  cls.routes={(*r['pe'],r['color']):r for r in cls.plan['routes']}
 def test_original_banks_and_routes_unchanged(self):
  base=Path(__file__).resolve().parents[1]/'evidence/layer-mlp-compile-029'
  manifest=json.loads((base/'source-manifest.json').read_text())['files']
  for n in ['gdn-bank-placement.json','worker-setups.json','gdn-compact-mlp.json','resident-gdn-routes.json']:
   self.assertEqual(self.files[n],verified_frozen(base/'source'/n,manifest[n]))
  old=retained_routes(self.files)+json.loads(self.files['resident-gdn-routes.json'])['routes']
  self.assertFalse(set(self.routes)&{(*r['pe'],r['color'])for r in old})
  self.assertEqual(len(self.routes),len(self.plan['routes']))
  self.assertFalse(self.plan['all_groups_routed']);self.assertEqual(self.plan['complete_routed_groups'],[0,1,2,14,15])
 def test_broadcast_words_and_bridge_boundaries(self):
  for plan in self.plan['plans']:
   covered=set()
   for domain in ['up','down']:
    color=plan['incoming'if domain=='up'else'outgoing'][0];source=tuple(plan['frontend'if domain=='up'else'bridge'])
    todo=[(source,'RAMP')];seen=set();sinks={}
    while todo:
     pe,rx=todo.pop();self.assertNotIn(pe,seen);seen.add(pe)
     r=self.routes[(*pe,color)];self.assertEqual(r['rx'],[rx])
     for d in r['tx']:
      if d=='RAMP':sinks[pe]=r
      else:todo.append((neighbor(pe,d),OPPOSITE[d]))
    expected={tuple(p)for p in plan['source_workers'if domain=='up'else'child_workers']}
    self.assertEqual(set(sinks),expected|({tuple(plan['bridge'])}if domain=='up'else set()))
    for pe in expected:
     self.assertNotIn(pe,covered);covered.add(pe);c=sinks[pe]['counter'];count=c['initial'];accepted=[]
     for token in range(2):
      for head in range(3):
       for k in range(387):
        if count<=c['maximum']:accepted.append((token,head,k))
        count=0 if count==c['limit']else count+1
     self.assertEqual(accepted,[(t,self.workers[pe]['head']%3,k)for t in range(2)for k in range(387)])
    if domain=='up':self.assertNotIn('counter',sinks[tuple(plan['bridge'])])
   self.assertEqual(covered,{pe for pe,w in self.workers.items()if w['head']//3==plan['group']})
 def test_actual_return_paths_and_weighted_debt(self):
  for plan in self.plan['plans']:
   values=Counter();debts=0
   for domain in ['down','up']:
    color=plan['incoming'if domain=='up'else'outgoing'][1];sink=tuple(plan['frontend'if domain=='up'else'bridge'])
    selected={tuple(p)for p in plan['source_workers'if domain=='up'else'child_workers']}
    # Recover the order from emitted RX/switch records, not plan path arrays.
    current=sink;next_rx=self.routes[(*current,color)]['rx'][0];sequence=[];visited={current}
    while next_rx!='RAMP':
     current=neighbor(current,next_rx);self.assertNotIn(current,visited);visited.add(current)
     r=self.routes[(*current,color)]
     if r['rx']==['RAMP']:
      sequence.append(current);next_rx=r['switch']['next_rx']if 'switch'in r else'RAMP'
      if current in selected and len(set(sequence)&selected)==len(selected)and domain=='down':break
     else:next_rx=r['rx'][0]
    self.assertEqual(set(sequence),selected|({tuple(plan['bridge'])}if domain=='up'else set()))
    advanced=set();frames=0
    for origin in sequence:
     pe=origin;arrival='RAMP';seen=set()
     while True:
      self.assertNotIn(pe,seen);seen.add(pe);r=self.routes[(*pe,color)]
      self.assertEqual([r['switch']['next_rx']]if pe in advanced else r['rx'],[arrival])
      d=r['tx'][0]
      if d=='RAMP':self.assertEqual(pe,sink);break
      pe=neighbor(pe,d);arrival=OPPOSITE[d]
     if origin in selected:
      w=self.workers[origin];values.update((w['head'],v)for v in range(w['first'],w['first']+w['columns']));frames+=w['columns']//2
      self.assertTrue(self.routes[(*origin,color)]['switch']['pop_on_advance']);advanced.add(origin)
      if domain=='up':debts+=1
     else:debts+=plan['child_markers']
    if domain=='down':self.assertEqual(frames,plan['child_frames']);self.assertEqual(len(selected),plan['child_markers'])
   self.assertEqual(values,Counter({(h,v):1 for h in range(3*plan['group'],3*plan['group']+3)for v in range(128)}))
   self.assertEqual(debts,len([w for w in self.workers.values()if w['head']//3==plan['group']]))
 def test_cohost_bindings_and_old_marker_abi(self):
  profiles={tuple(p['pe']):p for p in self.profiles}
  for plan in self.plan['plans']:
   p=profiles[tuple(plan['bridge'])];a=p['parameters'];self.assertEqual(a['bridge_frames'],plan['child_frames']);self.assertEqual(a['bridge_workers'],plan['child_markers'])
   if p['source']=='bridge_compact_layer_projection.csl':
    self.assertTrue(a['copy_transport']);self.assertFalse(a.get('mlp_sender',False));self.assertFalse(a.get('fusion_actor',False));self.assertEqual(a['bridge_reverse_queue'],5)
   front=self.files[profiles[tuple(plan['frontend'])]['source']]
   self.assertIn(b'task gdn_marker() void {@assert(gdn_markers>0);gdn_markers-=1;}',front)
   self.assertIn(b'@get_control_task_id(41)',front)
   for domain,key in [('incoming','source_workers'),('outgoing','child_workers')]:
    for pe in plan[key]:
     p=profiles[tuple(pe)];self.assertEqual(p['parameters']['gdn_input_color'],plan[domain][0]);self.assertEqual(p['parameters']['gdn_output_color'],plan[domain][1]);self.assertTrue(p['source'].startswith('resident_gdn_'))

if __name__=='__main__':unittest.main()
