"""Bounded synthetic observation faults; no original parameters, model, or SDK."""
import json
import hashlib
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'core'),str(ROOT/'examples/wp16/diagnostic')]
from checks import Checks,numeric_state
from qwen38.wp16_diag_layout import sequence,tile_snapshot

def fixture(tile):
    prefixes={stage+'_prefix_fp32':np.zeros((46,128,2),np.float64) for stage in ('gate','up')}
    prefixes['prefix_columns']=np.array([min(112*(i+1),5120) for i in range(46)],np.uint32)
    check=Checks({}, {}, prefixes);check.next_tile[0]=tile
    items=[x for x in sequence() if x['kind']=='copy' and x['pe']==0 and x['tile']==tile]
    weights=np.full(14338,0x3f80,np.uint32);weights[0]=0xa55a;weights[-1]=0x5aa5
    inputs=np.zeros(114,np.float32);inputs[0]=77.5;inputs[-1]=-88.25
    if tile==45:inputs[81:-1]=7.
    output=np.zeros(130,np.float32);output[0]=1234.5;output[-1]=-4321.25
    arrays=dict(weights_storage=weights,input_storage=inputs,output_storage=output,
        state=np.array(numeric_state(0,phase=1,tiles=tile+1,consumed=min(112*(tile+1),5120)),np.uint32),
        tile_guard_snapshot=np.array(tile_snapshot(0,tile),np.uint32),
        control=np.array([1,0,1,5120,tile,112*tile,min(112,5120-112*tile),0],np.uint32))
    for item in items:
        if item['direction']=='h2d':check.upload(item,arrays[item['name']])
    return check,[x for x in items if x['direction']=='d2h'],arrays

class TileObservationTests(unittest.TestCase):
    def test_complete_first_middle_last_groups_unlock_only_after_snapshot(self):
        for tile in (0,1,45):
            check,items,arrays=fixture(tile)
            for item in items[:-1]:
                check.receive(item,arrays[item['name']].copy())
                self.assertIn(0,check.pending_weights)
            check.receive(items[-1],arrays['tile_guard_snapshot'].copy())
            self.assertNotIn(0,check.pending_weights)
            self.assertEqual(check.next_tile[0],tile+1)
            self.assertEqual(check.reports['tile_groups'][0]['full_weight_payload_bitwise_D2H'],tile in (0,45))

    def test_early_overwrite_and_missing_observations_fail(self):
        check,items,arrays=fixture(1)
        upload=next(x for x in sequence() if x['kind']=='copy' and x['direction']=='h2d' and x['name']=='weights_storage' and x['pe']==0 and x['tile']==2)
        with self.assertRaisesRegex(ValueError,'overwrite'):check.upload(upload,arrays['weights_storage'])
        for omitted in ('input_storage','output_storage','state'):
            check,items,arrays=fixture(1)
            for item in items[:-1]:
                if item['name']!=omitted:check.receive(item,arrays[item['name']].copy())
            with self.assertRaisesRegex(ValueError,'All tile'):check.receive(items[-1],arrays['tile_guard_snapshot'].copy())
            self.assertIn(0,check.pending_weights)

    def test_every_snapshot_field_is_bound(self):
        for slot in range(12):
            check,items,arrays=fixture(1)
            for item in items[:-1]:check.receive(item,arrays[item['name']].copy())
            changed=arrays['tile_guard_snapshot'].copy();changed[slot]^=1
            with self.assertRaisesRegex(ValueError,'snapshot'):check.receive(items[-1],changed)
            self.assertIn(0,check.pending_weights)

    def test_invalid_source_output_and_unread_endpoint_weights_cannot_unlock(self):
        check,items,arrays=fixture(1)
        arrays['output_storage'][77]=1.
        for item in items[:-1]:
            if item['name']=='output_storage':
                with self.assertRaisesRegex(ValueError,'interval violations'):check.receive(item,arrays[item['name']].copy())
            else:check.receive(item,arrays[item['name']].copy())
        with self.assertRaisesRegex(ValueError,'All tile'):check.receive(items[-1],arrays['tile_guard_snapshot'].copy())
        check,items,arrays=fixture(0)
        for item in items[:-1]:
            if item['name']!='weights_storage':check.receive(item,arrays[item['name']].copy())
        with self.assertRaisesRegex(ValueError,'All tile'):check.receive(items[-1],arrays['tile_guard_snapshot'].copy())

    def test_schedule_exactly_matches_controller_reviewed_proposal(self):
        actual=hashlib.sha256(json.dumps(sequence(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(actual,'7954cc850d719bc7df7fa18ad483d2dcc2dbf9fe9533fdfec6d583bd49bbcb5b')

if __name__=='__main__':unittest.main()
