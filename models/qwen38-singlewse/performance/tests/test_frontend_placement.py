"""Original tile/state conservation and adversarial compact/native lowering."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial.compact_mixer import CompactMixerPlacement
from spatial.frontend_placement import plan_frontend_stage,lower_frontend_banks,audit_frontend_banks,FrontendAuxiliaryPlacement
from spatial.mixer_frontend_network import audit_frontend_network


class FrontendPlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        read=lambda p:json.loads(p.read_text());base=ROOT/'evidence/layer-mlp-compile-021/source'
        cls.region=next(r for r in read(base/'joint-stage.json')['regions'] if r['role']=='mix')
        cls.dialogue=read(base/'dialogue-bank-placement.json')
        profiles=[dict(pe=pe,source=c['source'],parameters=dict(c['parameters'])) for c in read(base/'profiles.json')['profiles'] for pe in c['pes']]
        cls.args=[cls.region,profiles,cls.dialogue,read(ROOT/'evidence/dialogue-bank-census-001/result.json'),
                  read(ROOT/'evidence/layer-backend-compile-026/source/profiles.json')['samples'],
                  read(ROOT/'evidence/layer-backend-compile-026/sram.json'),read(ROOT/'evidence/layer-backend-compile-027/sram.json')]
        cls.plan=plan_frontend_stage(*cls.args);cls.banks=lower_frontend_banks(cls.dialogue,cls.plan)
        cls.compact=CompactMixerPlacement(cls.plan['region'])
        cls.setups=[[r['pe'],r['setup']] for r in cls.compact.records.values() if r['pe']!=[107,145]]

    def test_all_original_rows_k_parts_and_scale_block_identities(self):
        originals=self.region['matrices'];seen=[set() for _ in originals];scale_ids={};data_words=0
        for rank,record in self.compact.records.items():
            intervals=[];s=record['setup']
            for mi,m in enumerate(originals):
                base,count,first,stride,key,parts,bf16,rows=s[8*mi:8*mi+8]
                self.assertEqual((parts,bf16,rows),(m['shape'][1]//128,int(m['dtype']=='BF16'),m['tile_shape'][0]))
                if count:intervals.append((base,base+64*count))
                for i in range(count):
                    row=first+i*stride;tile=(row//rows)*parts+key
                    self.assertNotIn(tile,seen[mi]);seen[mi].add(tile)
                    address=self.compact.tile(mi,tile)
                    self.assertEqual((address['pe'],address['byte_offset'],address['bytes']),(record['pe'],4*(base+64*i),256))
                    if not bf16:
                        scale=s[40+mi]+row//128-first//128
                        self.assertEqual(address['scale']['byte_offset'],4*scale)
                        ident=(mi,row//128,key);place=(rank,scale)
                        self.assertEqual(scale_ids.setdefault(place,ident),ident)
                data_words+=64*count
            cursor=0
            for first,last in sorted(intervals):self.assertEqual(first,cursor);cursor=last
            self.assertEqual(cursor,record['data_words'])
            self.assertLessEqual(record['bytes']-4*cursor,64)
        self.assertEqual([s for s in map(len,seen)],[m['tiles'] for m in originals])
        self.assertEqual(data_words*4,454400*256)
        self.assertEqual(len(scale_ids)*4,450560*4-self.banks['scale_alias_bytes'])

    def test_complete_single_dialogue_with_two_work_slots_survives(self):
        result=audit_frontend_banks(self.dialogue,self.plan,self.banks)
        self.assertEqual(result['pages'],28605)
        self.assertEqual(self.banks['allocated_bytes'],392013504)
        self.assertEqual(self.banks['scale_alias_bytes'],1732544)
        self.assertEqual(self.banks['retained_pages_changing_pe'],3301)
        resolver=FrontendAuxiliaryPlacement(self.dialogue,self.plan,self.banks)
        for obj in self.banks['objects']:
            for page in range(obj['page_start'],obj['page_start']+obj['retained_pages']):self.assertEqual(resolver.owner(page)['bytes'],128)
            if obj['dropped_pages']:
                with self.assertRaises(ValueError):resolver.owner(obj['page_start']+obj['retained_pages'])
        self.assertFalse(self.banks['source_alias_values_verified'])
        self.assertFalse(self.banks['compiled'])

    def test_state_holes_overlaps_and_prefix_changes_are_rejected(self):
        for change in ('missing','duplicate','physical_alias','prefix','limit','second_context'):
            bank=deepcopy(self.banks);plan=deepcopy(self.plan)
            if change=='missing':bank['spans'].pop()
            if change=='duplicate':bank['spans'].insert(0,deepcopy(bank['spans'][0]))
            if change=='physical_alias':bank['spans'][0]['byte_offset']-=128
            if change=='prefix':bank['bank_prefixes'][0]['bytes']-=4
            if change=='limit':plan['limits'][0]['bytes']=0
            if change=='second_context':bank['objects'][0]['retained_pages']+=1
            with self.subTest(change=change),self.assertRaises(ValueError):audit_frontend_banks(self.dialogue,plan,bank)

    def test_native_network_counts_foreign_frames_until_full_drain(self):
        plan=self.plan['network'];audit=audit_frontend_network(plan,self.setups)
        self.assertEqual((audit['projected_packets'],audit['original_projection_values'],len(plan['mergers'])),(8288,16480,168))
        self.assertEqual(sum(plan['consumer_projected_packets']),29367)
        self.assertTrue(all(n>518 for n in plan['consumer_projected_packets']))
        self.assertTrue(plan['batched_source_rows'])
        self.assertTrue(all('filter_tag' not in r for r in plan['routes']))

    def test_native_route_misdelivery_and_early_drain_are_rejected(self):
        for change in ('early_drain','foreign_tag','source_group','missing','merge','duplicate'):
            plan=deepcopy(self.plan['network'])
            if change=='early_drain':plan['consumer_projected_packets'][0]=518
            if change=='foreign_tag':plan['routes'][0]['filter_tag']=0
            if change=='source_group':plan['producers'][0]['groups'].append(15)
            if change=='missing':plan['routes'].pop(0)
            if change=='merge':plan['mergers'][0]['horizontal']=[]
            if change=='duplicate':plan['routes'].append(deepcopy(plan['routes'][0]))
            with self.subTest(change=change),self.assertRaises(ValueError):audit_frontend_network(plan,self.setups)

    def test_unmeasured_calibration_and_overbudget_reader_are_rejected(self):
        for change in ('missing_elf','stack','code'):
            args=deepcopy(self.args)
            if change=='missing_elf':args[5]['records'].pop()
            if change=='stack':args[6]['records'][0]['stack_allowance_bytes']=0
            if change=='code':args[6]['records'][0]['low_section_end']+=256
            with self.subTest(change=change),self.assertRaises(ValueError):plan_frontend_stage(*args)


if __name__=='__main__':unittest.main()
