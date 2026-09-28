"""Original scale-block identities, ragged K ownership and retained page shifts."""
import json,sys,unittest
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.compact_mlp import CompactMlpPlacement, CompactMlpAuxiliaryPlacement, lower_mlp_banks, audit_mlp_banks
from spatial.layer_schedule import tile_owner
from tools.build_compact_mlp_stage import build

class CompactMlpTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.source=ROOT/'evidence/layer-mlp-compile-022/source'
  cls.stage=json.loads((cls.source/'frontend-stage.json').read_text());cls.compact=CompactMlpPlacement(cls.stage)
  cls.previous=json.loads((cls.source/'frontend-bank-placement.json').read_text());cls.plan=lower_mlp_banks(cls.previous,cls.compact)

 def test_all_original_descriptors_and_auxiliary_suffixes_survive(self):
  old={tuple(pe):v for pe,v in json.loads((self.source/'worker-setups.json').read_text())}
  self.assertEqual({pe:r['original_setup'] for pe,r in self.compact.records.items()},old)
  self.assertEqual(len(old),7833)
  prefixes={tuple(p['pe']):p['bytes'] for p in self.previous['bank_prefixes']}
  new={tuple(p['pe']):p['bytes'] for p in self.plan['bank_prefixes']}
  for pe,r in self.compact.records.items():
   self.assertEqual(prefixes[pe]-r['original_bytes'],new[pe]-r['bytes'])
   self.assertGreaterEqual(new[pe]-r['bytes'],0)
  self.assertEqual(self.plan['scale_alias_bytes'],3912128)

 def test_block_edges_and_ragged_k_keep_global_scale_identity(self):
  aliases={};observed_parts=set();checked=0
  for role,region in self.compact.regions.items():
   for mi,m in enumerate(region['matrices']):
    rows,columns=m['tile_shape']
    for group in self.compact.groups[role]:
     first=group['output_start'];stop=first+group['output_count'];rowset={first,stop-1}
     rowset.update(r for r in range(first,stop) if (r*rows)%128 in (0,128-rows))
     for row in sorted(rowset):
      for key in {0,group['workers']-1,m['k_blocks']-1}:
       tile=row*m['k_blocks']+key;old=tile_owner(region,mi,tile);new=self.compact.tile(role,mi,tile)
       self.assertEqual(old['pe'],new['pe']);record=self.compact.records[tuple(old['pe'])];observed_parts.add(record['parts'])
       identity=(role,mi,row*rows//128,key*columns//128)
       address=(*new['pe'],new['scale']['byte_offset'])
       if address in aliases:self.assertEqual(identity,aliases[address])
       aliases[address]=identity
       self.assertLessEqual(new['byte_offset']+256,record['bytes'])
       self.assertLessEqual(new['scale']['byte_offset']+4,record['bytes']);checked+=1
  self.assertEqual(observed_parts,{1,2});self.assertEqual(checked,1519)

 def test_every_retained_page_moves_only_by_local_prefix_savings(self):
  resolver=CompactMlpAuxiliaryPlacement(self.previous,self.compact,self.plan);count=0
  for s in self.previous['spans']:
   r=self.compact.records.get(tuple(s['pe']));shift=r['original_bytes']-r['bytes'] if r else 0
   for page in range(s['page_start'],s['page_start']+s['pages']):
    self.assertEqual(resolver.owner(page),dict(pe=s['pe'],byte_offset=s['byte_offset']+128*(page-s['page_start'])-shift,bytes=128));count+=1
  self.assertEqual(count,28605)
  for page in (-1,1204,26260,53661):
   with self.assertRaises(ValueError):resolver.owner(page)
  broken=deepcopy(self.plan);broken['spans'][0]['byte_offset']+=4
  with self.assertRaises(ValueError):audit_mlp_banks(self.previous,self.compact,broken)
  broken=deepcopy(self.plan);broken['work_slots']=1
  with self.assertRaises(ValueError):audit_mlp_banks(self.previous,self.compact,broken)

 def test_complete_builder_preserves_every_existing_route_and_neural_module(self):
  files,width,height,profiles=build();self.assertEqual((width,height,len(profiles)),(78,146,11388))
  original=(self.source/'layout.csl').read_bytes()
  self.assertEqual(files['layout.csl'].split(b' const route_0=',1)[1].rstrip(),original.split(b' const route_0=',1)[1].rstrip())
  for p in self.source.glob('*.csl'):
   if p.name!='layout.csl':self.assertEqual(files[p.name],p.read_bytes(),p.name)
  old={tuple(pe):(c['source'],c['parameters']) for c in json.loads((self.source/'profiles.json').read_text())['profiles'] for pe in c['pes']}
  for p in profiles:
   pe=tuple(p['pe']);source,params=old[pe]
   if pe not in self.compact.records:self.assertEqual((p['source'],p['parameters']),(source,params))
   else:
    before=dict(params);after=dict(p['parameters']);before.pop('bank_words');after.pop('bank_words');self.assertEqual(before,after)

if __name__=='__main__':unittest.main()
