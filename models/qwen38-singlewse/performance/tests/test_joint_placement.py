"""Original-dimension row coverage, disjoint state, and corrupted-map rejection."""
import json,sys,unittest
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.joint_placement import lower_joint_banks,audit_joint_banks,JointAuxiliaryPlacement
from spatial.mixer_projection import lower_mixer_projections


class JointPlacementTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  e=ROOT/'evidence'
  stage=next(s for s in json.loads((e/'layer-native-schedule-002/layer-schedule.json').read_text())['stages'] if s['id']=='layer_00')
  cls.original=next(r for r in stage['regions'] if r['role']=='mix')
  profiles=[dict(pe=pe,source=c['source'],parameters=dict(c['parameters'])) for c in json.loads((e/'layer-mlp-compile-015/source/profiles.json').read_text())['profiles'] for pe in c['pes']]
  cls.region,cls.plan=lower_joint_banks(stage,profiles,json.loads((e/'joint-placement-calibration-001.json').read_text()),json.loads((e/'device-control-calibration-001.json').read_text()))
  cls.mixer=lower_mixer_projections(cls.region,stage['request_controller']['pe'])

 def test_routed_control_space_preserves_all_original_banks_and_state(self):
  e=ROOT/'evidence'
  stage=next(s for s in json.loads((e/'layer-native-schedule-002/layer-schedule.json').read_text())['stages'] if s['id']=='layer_00')
  profiles=[dict(pe=pe,source=c['source'],parameters=dict(c['parameters'])) for c in json.loads((e/'layer-mlp-compile-015/source/profiles.json').read_text())['profiles'] for pe in c['pes']]
  region,plan=lower_joint_banks(stage,profiles,json.loads((e/'joint-placement-calibration-001.json').read_text()),json.loads((e/'device-control-calibration-001.json').read_text()),mix_extra_reserve=256)
  self.assertEqual(plan['original_allocated_bytes'],396953216)
  self.assertEqual(plan['original_pages'],53661)
  self.assertEqual(plan['extra_bytes_reserved'],256)
  self.assertEqual(plan['mixer_additional_code_reserve'],256)
  self.assertGreaterEqual(plan['unused_page_capacity'],0)
  self.assertEqual(lower_mixer_projections(region,stage['request_controller']['pe'])['audit']['original_tiles_checked'],454400)
  resolver=JointAuxiliaryPlacement(self.original,region,plan)
  self.assertEqual(len({(*resolver.owner(p)['pe'],resolver.owner(p)['byte_offset']) for p in range(53661)}),53661)

 def test_every_original_qkv_row_has_exactly_original_40_k_blocks(self):
  coverage=bytearray(10240*40)
  for worker in self.mixer['workers']:
   d=worker['descriptors'][0]
   for i in range(d['iterations']):
    for row in range(d['first_row']+i*d['row_stride'],d['first_row']+i*d['row_stride']+2):
     index=row*40+d['key_index'];self.assertEqual(coverage[index],0);coverage[index]=1
  self.assertEqual(coverage,bytearray([1])*len(coverage))
  self.assertEqual(self.mixer['audit']['original_tiles_checked'],454400)

 def test_all_original_pages_and_two_full_gdn_states_have_unique_addresses(self):
  resolver=JointAuxiliaryPlacement(self.original,self.region,self.plan)
  addresses=[(*resolver.owner(p)['pe'],resolver.owner(p)['byte_offset']) for p in range(53661)]
  self.assertEqual(len(set(addresses)),53661)
  states=[]
  for request in range(2):
   for head in range(48):
    for key in range(32):
     for value in range(16):
      p=resolver.gdn_owner(request,head,key,value);states.append((*p['pe'],p['byte_offset']))
  self.assertEqual(len(set(states)),49152)
  self.assertEqual(self.plan['original_allocated_bytes'],396953216)
  self.assertGreater(self.plan['auxiliary_pages_outside_mixer'],0)

 def test_missing_page_and_physical_alias_are_rejected(self):
  p=deepcopy(self.plan);p['spans'].pop(0)
  with self.assertRaisesRegex(ValueError,'Missing or duplicated'):audit_joint_banks(self.original,self.region,p)
  p=deepcopy(self.plan);p['spans'][1]['pe']=p['spans'][0]['pe'];p['spans'][1]['byte_offset']=p['spans'][0]['byte_offset']
  with self.assertRaises(ValueError):audit_joint_banks(self.original,self.region,p)

 def test_missing_qkv_rows_and_shortened_k_group_are_rejected(self):
  for field,delta in [('output_start',1),('workers',-1)]:
   r=deepcopy(self.region);r['matrices'][0]['group_partitions'][1][field]+=delta
   with self.assertRaises(ValueError):audit_joint_banks(self.original,r,self.plan)
   with self.assertRaises(ValueError):lower_mixer_projections(r,[107,145])

if __name__=='__main__':unittest.main()
