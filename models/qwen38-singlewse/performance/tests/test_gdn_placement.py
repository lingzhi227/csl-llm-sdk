"""Reject lost original rows, state, pages and changed cohost lease contracts."""
from copy import deepcopy
import json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.gdn_placement import plan,audit,STATE_FIRST,GdnBankPlacement,refine_from_census
from spatial.gdn_columns import compose_controlled_port
from spatial.layer_mlp_network import build_mlp_network,worker_profiles
from spatial.layer_fused_routes import build_fused_routes

class GdnPlacementTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=ROOT/'evidence/layer-mlp-compile-024/source';read=lambda n:json.loads((cls.root/n).read_text())
  cls.stage=read('frontend-stage.json');cls.banks=read('mlp-bank-placement.json')
  cls.profiles=[dict(pe=pe,source=c['source'],parameters=c['parameters']) for c in read('profiles.json')['profiles'] for pe in c['pes']]
  cls.census=json.loads((ROOT/'evidence/compact-mlp-census-001/result.json').read_text())
  cls.plan=plan(cls.stage,cls.banks,cls.census,cls.profiles)

 def test_all_original_state_coordinates_are_bijective(self):
  seen=bytearray(48*128*128);oldwords=bytearray(len(seen));count=0
  for w in self.plan['workers']:
   for key in range(128):
    for col in range(w['columns']):
     value=w['first']+col;head=w['head']
     canonical=(head*128+key)*128+value
     oldpage=STATE_FIRST+head*512+(key//4)*16+value//8
     oldword=(oldpage-STATE_FIRST)*32+(key%4)*8+value%8
     # Independent inverse of the original4x8 bank tiles.
     page,word=divmod(oldword,32);h,local=divmod(page,512);kb,vb=divmod(local,16);kr,vr=divmod(word,8)
     self.assertEqual((h,4*kb+kr,8*vb+vr),(head,key,value))
     self.assertFalse(seen[canonical]);self.assertFalse(oldwords[oldword]);seen[canonical]=oldwords[oldword]=1;count+=1
  self.assertEqual(count,786432);self.assertTrue(all(seen));self.assertTrue(all(oldwords))

 def test_intact_namespaces_and_matrix_extent(self):
  self.assertEqual(self.plan['audit']['original_mlp_tiles'],1044480)
  self.assertEqual(self.plan['auxiliary_pages'],7085)
  self.assertEqual(sum(self.plan['counts']['gate_up']),2176);self.assertEqual(sum(self.plan['counts']['down']),1280)
  self.assertEqual(sum(w['columns'] for w in self.plan['workers']),6144)
  self.assertEqual(self.plan['allocated_bytes'],388094976)
  self.assertFalse(self.plan['compiled']);self.assertFalse(self.plan['routed'])
  for field in ('conversations','work_slots','recurrent_elements'):
   broken=deepcopy(self.plan);broken[field]+=1
   with self.assertRaises(ValueError):audit(self.stage,self.banks,broken)

 def test_reject_missing_rows_pages_state_and_overlap(self):
  for kind in ('row','page','state','bank'):
   broken=deepcopy(self.plan)
   if kind=='row':next(r for r in broken['stage']['regions'] if r['role']=='gate_up')['matrices'][0]['group_partitions'][0]['output_count']-=1
   if kind=='page':broken['spans'][0]['pages']-=1
   if kind=='state':broken['workers'][0]['first']+=1
   if kind=='bank':broken['bank_prefixes'][0]['bytes']+=4
   with self.assertRaises(ValueError):audit(self.stage,self.banks,broken)

 def test_rebalanced_native_graph_delivers_every_original_output(self):
  stage=self.plan['stage'];gate=next(r for r in stage['regions'] if r['role']=='gate_up')
  fused=build_fused_routes(gate);network=build_mlp_network(stage,fused,True,True)
  self.assertTrue(network['audit']['passed']);self.assertEqual(len(worker_profiles(stage,fused,network)),7833)
  self.assertEqual(network['controller_transport']['frame_words'],277)
  self.assertEqual(sorted(q['index'] for q in network['grant_schedule'] if q['kind']==1),list(range(136)))
  # Every output sink is an exact disjoint contiguous piece of the5120 outputs.
  rows=[r for s in network['down_sinks'] for r in range(s['first_row'],s['first_row']+s['rows'])]
  self.assertEqual(sorted(rows),list(range(5120)))

 def test_actual_census_refinement_moves_only_ten_auxiliary_pages(self):
  census=json.loads((ROOT/'evidence/gdn-cohost-census-001.json').read_text())
  refined=refine_from_census(self.stage,self.banks,self.plan,census)
  self.assertEqual(refined['workers'],self.plan['workers']);self.assertEqual(refined['stage'],self.plan['stage'])
  self.assertEqual(refined['bank_prefixes'],self.plan['bank_prefixes'])
  self.assertEqual(refined['allocated_bytes'],self.plan['allocated_bytes'])
  def owners(p):return {(s['namespace'],s['page_start']+i):s['pe'] for s in p['spans'] for i in range(s['pages'])}
  before=owners(self.plan);after=owners(refined);self.assertEqual(before.keys(),after.keys())
  self.assertEqual(sum(before[k]!=after[k] for k in before),10)
  self.assertEqual(refined['reservation_refinement']['original_overflow_pes'],5)
  self.assertEqual(len(refined['reservation_refinement']['changed_bank_pes']),9)
  self.assertTrue(all(p['projected_sram']<=48128 for p in refined['reservation_refinement']['changed_bank_pes']))
  bad=deepcopy(census);bad['cells'][0][2]=100000
  with self.assertRaises(ValueError):refine_from_census(self.stage,self.banks,self.plan,bad)

 def test_resolver_does_not_flatten_strided_recurrent_pages(self):
  resolver=GdnBankPlacement(self.stage,self.banks,self.plan)
  widths=set();span_counts=set()
  for worker in self.plan['workers']:
   widths.add(worker['columns'])
   for key in (0,3,4,127):
    for value in (worker['first'],worker['first']+worker['columns']-1):
     page=STATE_FIRST+worker['head']*512+(key//4)*16+value//8
     spans=resolver.recurrent_spans(page);span_counts.add(len(spans))
     self.assertEqual(sum(s['bytes'] for s in spans),128)
     word=resolver.state_word(worker['head'],key,value)
     self.assertTrue(any(word['pe']==s['pe'] and s['byte_offset']<=word['byte_offset']<s['byte_offset']+s['bytes'] for s in spans))
     with self.assertRaises(ValueError):resolver.auxiliary_owner('mix',page)
  self.assertEqual(widths,{8,32});self.assertEqual(span_counts,{1,4})
  for span in self.plan['spans']:
   self.assertEqual(resolver.auxiliary_owner(span['namespace'],span['page_start']),dict(pe=span['pe'],byte_offset=span['byte_offset'],bytes=128))
  for page in (-1,1204,26260,53661):
   with self.assertRaises(ValueError):resolver.auxiliary_owner('mix',page)
  broken=deepcopy(self.plan);broken['source_objects']['mix'][0]['id']='wrong tensor'
  with self.assertRaises(ValueError):GdnBankPlacement(self.stage,self.banks,broken)

 def test_explicit_reset_and_admit_preserve_startup_and_lease_guards(self):
  for role in ('compact_layer_projection.csl','device_mixer_layer_mlp_standby.csl'):
   source=(self.root/role).read_bytes();raw,proof=compose_controlled_port(source,role)
   self.assertIn(b'fn start() void {sys.unblock_cmd_stream();}',raw)
   self.assertIn(b'@assert(gdn_local_idle()',raw);self.assertFalse(proof['startup_resets_state'])
   self.assertTrue(proof['local_dma_completion_is_not_downstream_retirement'])
   self.assertFalse(proof['neural_executed'])

if __name__=='__main__':unittest.main()
