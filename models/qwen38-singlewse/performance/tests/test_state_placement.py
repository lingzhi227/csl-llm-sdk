"""All original recurrent pages remain addressable after fusion SRAM placement."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial.state_placement import StatePlacement
from spatial.layer_schedule import gdn_state_owner


class StatePlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan=json.loads((ROOT/'evidence/layer-native-schedule-002/layer-schedule.json').read_text())
        stage=next(s for s in plan['stages'] if s['id']=='layer_00')
        cls.region=next(r for r in stage['regions'] if r['role']=='mix')
        # Actual immutable SRAM-admitted original bank placement, no candidate
        # generated expected addresses: all800 original logical pages are bound.
        source=ROOT/'evidence/layer-mlp-compile-012/source'
        profiles=json.loads((source/'profiles.json').read_text())['profiles']
        cls.profiles=[dict(pe=pe,parameters=p['parameters']) for p in profiles for pe in p['pes']]
        cls.pages=json.loads((source/'state-page-remap.json').read_text())['pages']

    def test_all_two_request_pages_and_every_scalar_remain_disjoint(self):
        placement=StatePlacement(self.region,self.profiles,self.pages)
        addresses=set();moved=0
        for request in range(2):
            for head in range(48):
                for key in range(32):
                    for value in range(16):
                        old=gdn_state_owner(self.region,request,head,key,value)
                        new=placement.gdn_owner(request,head,key,value)
                        address=(*new['pe'],new['byte_offset'])
                        self.assertNotIn(address,addresses);addresses.add(address)
                        moved+=address!=(*old['pe'],old['byte_offset'])
                        for field in ('state','request','head','key_start','value_start','shape','bytes'):
                            self.assertEqual(new[field],old[field])
                        self.assertEqual(new['shape'],[4,8])
        self.assertEqual(len(addresses),49152)
        self.assertEqual(moved,800)

    def test_missing_duplicate_foreign_and_overlapping_moves_rejected(self):
        for change in ('missing','duplicate','source','destination','tensor','logical'):
            with self.subTest(change=change):
                pages=copy.deepcopy(self.pages)
                if change=='missing':pages.pop()
                elif change=='duplicate':pages[1]=copy.deepcopy(pages[0])
                elif change=='source':pages[0]['source_byte_offset']+=128
                elif change=='destination':pages[0]['destination_byte_offset']=0
                elif change=='tensor':pages[0]['tensor']='recurrent.foreign'
                else:pages[0]['tensor_byte_offset']+=128
                with self.assertRaises(ValueError):StatePlacement(self.region,self.profiles,pages)

    def test_request_and_original_shape_bounds_still_enforced(self):
        placement=StatePlacement(self.region,self.profiles,self.pages)
        for coordinate in ((2,0,0,0),(0,48,0,0),(0,0,32,0),(0,0,0,16),(-1,0,0,0)):
            with self.assertRaises(ValueError):placement.gdn_owner(*coordinate)


if __name__=='__main__':unittest.main()
