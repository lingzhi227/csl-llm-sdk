"""Independent ownership and stream-window checks for connected qualification."""
import ast,json,unittest
from collections import Counter
from pathlib import Path
from tools.stage_gdn_fusion_sim import build
ROOT=Path(__file__).resolve().parents[1]


class FusionGraphTests(unittest.TestCase):
 def test_device_timing_wrap_and_order(self):
  import numpy as np
  source=ROOT/'probes/gdn_fusion/run.py';tree=ast.parse(source.read_text())
  fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='timing_record')
  namespace={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
  def words(values,base=0):
   values=(np.asarray(values,np.uint64)+np.uint64(base))&np.uint64((1<<48)-1)
   return np.stack([(values>>np.uint64(shift))&65535 for shift in (0,16,32)],axis=-1).reshape(len(values),-1)
  front=words([[10,20,30,50,60,70,80]],(1<<48)-15)
  worker=words([[1,2,3,7,8,10,12,13]],(1<<48)-4)
  result=namespace['timing_record'](front,worker)
  self.assertEqual(result['frontend_first_launch_to_drain'],[70])
  self.assertEqual(result['frontend_head_launch_to_gated_output'],[[40,40,40]])
  self.assertEqual(result['worker_key_update_max'],4)
  self.assertEqual(result['worker_query_reduce_max'],2)
  with self.assertRaises(ValueError):namespace['timing_record'](front,words([[1,2,3,7,6,10,12,13]]))

 def test_complete_original_ownership_and_diagnostic_boundary(self):
  files=build();plan=json.loads(files['probe-plan.json'])
  original=json.loads((ROOT/'evidence/layer-mlp-compile-027/source/gdn-bank-placement.json').read_text())['workers']
  expected={tuple(w['pe']):w for w in original};seen=Counter()
  self.assertEqual(plan['application'],[160,5]);self.assertEqual(len(plan['workers']),753)
  for w in plan['workers']:
   self.assertEqual({k:v for k,v in w.items() if k!='diagnostic_pe'},expected[tuple(w['pe'])])
   self.assertEqual(w['diagnostic_pe'][0]//10,w['head']//3)
   seen.update((w['head'],v) for v in range(w['first'],w['first']+w['columns']))
  self.assertEqual(seen,Counter({(h,v):1 for h in range(48) for v in range(128)}))
  self.assertFalse(plan['full_bank_admission']);self.assertFalse(plan['host_injected_recurrent_results'])
  self.assertEqual(files['layout.csl'].count(b'@set_tile_code('),800)
  for name,raw in files.items():
   if name.endswith('.py'):ast.parse(raw)

 def test_return_markers_pop_only_on_local_switch_advance(self):
  # Each source consumes its ADV once. Ancestors already receive from child;
  # popping on every hop destroys the marker's control entry point on long paths.
  files=build(True);layout=files['layout.csl']
  self.assertEqual(layout.count(b'.pop_mode=.{.pop_on_advance=true}'),39)
  self.assertNotIn(b'always_pop',layout)
  self.assertIn(b'gdn_control.opcode.SWITCH_ADV',files['worker.csl'])
  self.assertIn(b'clear_current_position',files['worker.csl'])

 def test_counter_windows_retain_each_head_across_invocations(self):
  # WSE counter filters compare the current inclusive count, then increment.
  # A full group contains three fixed-length packets. No CE discards data.
  for head in range(3):
   count=(1161-387*head)%1161;accepted=[]
   for token in range(7):
    for h in range(3):
     for word in range(387):
      if count<=386:accepted.append((token,h,word))
      count=(count+1)%1161
   self.assertEqual(accepted,[(t,head,w) for t in range(7) for w in range(387)])

 def test_smoke_uses_all_three_original_heads_and_same_kernels(self):
  full=build();small=build(True);plan=json.loads(small['probe-plan.json'])
  self.assertEqual(plan['application'],[10,5]);self.assertEqual({w['head'] for w in plan['workers']},{0,1,2})
  self.assertEqual(Counter(w['columns'] for w in plan['workers']),{8:36,32:3})
  for name,raw in full.items():
   if name not in ('layout.csl','probe-plan.json'):self.assertEqual(raw,small[name],name)


if __name__=='__main__':unittest.main()
