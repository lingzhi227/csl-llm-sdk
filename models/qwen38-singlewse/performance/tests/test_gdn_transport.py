"""Exact original extents, producer ABI and transport harness admission."""
import ast,json
from collections import Counter
from pathlib import Path
import unittest
from spatial.gdn_columns import paired_port,compose_frontend_packet,compose_port
from tools.stage_gdn_columns_sim import build
ROOT=Path(__file__).resolve().parents[1]

class GdnTransportTests(unittest.TestCase):
 def test_paired_frame_rejects_odd_slices_and_preserves_math(self):
  raw=paired_port()
  self.assertIn(b'gdn_first%2==0 and gdn_columns%2==0',raw)
  self.assertIn(b'frame[2]=2;',raw);self.assertIn(b'gdn_output+=2;',raw)
  original=(ROOT/'runtime/gdn_columns_port.cslpart').read_bytes()
  for line in original.splitlines():
   if b'gdn_math.' in line:self.assertIn(line,raw)
 def test_frontend_changes_packet_stores_only(self):
  original=(ROOT/'csl/mixer_frontend_native.csl').read_bytes();changed,proof=compose_frontend_packet(original)
  a=original.index(b'fn prepare(');b=original.index(b'fn packet_consumed(')
  self.assertEqual(original[:a],changed[:a]);self.assertEqual(original[b:],changed[changed.index(b'fn packet_consumed('):])
  self.assertEqual(proof['fields'],['token','decay','beta','V128','K128','Q128'])
  with self.assertRaises(ValueError):compose_frontend_packet(changed)
 def test_all_original_slices_and_only_new_harness_colors(self):
  files=build();plan=json.loads(files['probe-plan.json']);workers=plan['workers'];seen=Counter()
  self.assertEqual(plan['application'],[251,6]);self.assertEqual(len(workers),753)
  self.assertEqual(Counter(w['columns'] for w in workers),{8:748,32:5})
  for w in workers:seen.update((w['head'],v) for v in range(w['first'],w['first']+w['columns']))
  self.assertEqual(seen,Counter({(h,v):1 for h in range(48) for v in range(128)}))
  text=files['layout.csl'];self.assertEqual(text.count(b'@set_tile_code('),1506)
  self.assertNotIn(b'@get_color(22)',text);self.assertNotIn(b'@get_color(21)',text)
  self.assertEqual(text.count(b'@get_color(3)'),1506);self.assertEqual(text.count(b'@get_color(4)'),1506)
  self.assertNotIn(b'PAIRED_GDN_PORT',files['worker.csl']);self.assertNotIn(b'GDN_WORKSPACE',files['worker.csl'])
  for n,b in files.items():
   if n.endswith('.py'):ast.parse(b)
 def test_smoke_only_changes_geometry_and_selected_original_slices(self):
  full=build();smoke=build(True)
  for name in full:
   if name not in ('layout.csl','probe-plan.json'):self.assertEqual(full[name],smoke[name],name)
  all_workers=json.loads(full['probe-plan.json'])['workers'];small=json.loads(smoke['probe-plan.json'])
  self.assertEqual(small['application'],[3,2]);self.assertFalse(small['full_original_value_coverage'])
  self.assertEqual(sorted(w['columns'] for w in small['workers']),[8,8,32])
  self.assertTrue(all(w in all_workers for w in small['workers']))
  self.assertEqual(smoke['layout.csl'].count(b'@set_tile_code('),6)

if __name__=='__main__':unittest.main()
