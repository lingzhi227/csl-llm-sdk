"""Small connected identity, event, ownership and exact schedule fixtures."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from qwen38.wp16_reuse_layout import ARRAYS,sequence,budget
from qwen38.wp16_reuse_protocol import (packet,acknowledgement,verify_packet,verify_ack,key,
    identity,numeric_state,final_numeric_state,mlp_state,tile_snapshot,weight_snapshot,
    lifecycle_state,completed_handoff_state)

class ConnectedProtocolTests(unittest.TestCase):
    def test_exact_approved_schedule_and_no_consumer_H2D(self):
        path=ROOT/'local/wp16/connected-reuse-proposal-ledger.json'
        if not path.exists():path=ROOT/'qualification/connected-reuse-proposal-ledger.json'
        original=json.loads(path.read_text())['operations'];corrected=[];changes=0
        for item in original:
            item=dict(item)
            if item.get('direction')=='d2h' and item['name']=='timing' and item['pe']<2:
                self.assertEqual((item['count'],item['physical_host_bytes'],item['native_device_bytes']),(276,1104,552))
                item.update(count=6,physical_host_bytes=24,native_device_bytes=12);changes+=1
            corrected.append(item)
        self.assertEqual(changes,12);self.assertEqual(sequence(),corrected)
        plan=budget();self.assertEqual((plan['physical_operations'],plan['copy_calls'],plan['launch_calls']),(696,647,49))
        for item in sequence():
            if item.get('direction')=='h2d':
                self.assertFalse(item['name'].startswith('mlp_'))
                self.assertFalse(item['name']=='input_storage' and item['pe']>=2)

    def test_timestamp_capacity_covers_every_live_access(self):
        self.assertEqual(ARRAYS['timing'],(16,(6,6,6,12)))
        for pe in range(4):
            count=ARRAYS['timing'][1][pe]
            writes=[tile*6+i+boundary for tile in range(2 if pe==3 else 1) for i in range(3) for boundary in (0,3)]
            self.assertEqual(sorted(writes),list(range(count)))
        plan=budget()
        self.assertEqual((plan['host_bytes'],plan['native_bytes'],plan['raw_D2H_bytes']),(978168,539652,660392))
        self.assertEqual(plan['declared_buffer_bytes_per_PE'],[31368,31368,4444,19158])

    def test_all_three_epoch_keys_are_distinct_and_wrong_input_rejected(self):
        keys=[key(g,i,c,'down_tile_readback','output_storage',3,0) for g,i,c in (identity(g) for g in (1,2,3))]
        self.assertEqual(len(set(keys)),3)
        for g in (1,2,3):
            with self.assertRaises(ValueError):key(g,999,g,'phase','state',0)
            with self.assertRaises(ValueError):key(g,identity(g)[1],g+1,'phase','state',0)

    def test_stale_version_epoch_scope_and_bad_payload_frames(self):
        body=[0x3f80]*128
        for g in (1,2,3):
            for edge in (0,1,2):
                frame=packet(g,edge,body);verify_packet(g,edge,frame)
                ack=acknowledgement(g,edge);verify_ack(g,edge,ack)
                self.assertEqual((len(frame),len(ack)),(144,17))
                for index in (0,1,2,*range(3,15),143):
                    changed=list(frame);changed[index]^=1
                    with self.assertRaises(ValueError):verify_packet(g,edge,changed)
                for index in range(17):
                    changed=list(ack);changed[index]^=1
                    with self.assertRaises(ValueError):verify_ack(g,edge,changed)
        with self.assertRaises(ValueError):verify_packet(2,0,packet(1,0,body))
        for bad in (0x10000,0x7f80,0x7fc0):
            changed=body.copy();changed[12]=bad
            with self.assertRaises(ValueError):packet(1,0,changed)

    def test_generation_counters_and_reset_commit_release_vectors(self):
        for g in (1,2,3):
            for pe in range(4):
                reset=lifecycle_state(g,pe,stage='reset');retained=lifecycle_state(g,pe,stage='retention');released=lifecycle_state(g,pe,stage='released')
                self.assertEqual(reset[12:15],[g,g,g-1]);self.assertEqual(released[12:15],[g,g,g])
                self.assertEqual(reset[16:18],[8191,g]);self.assertEqual(released[29:31],[2*g-1,2*g])
                self.assertEqual(reset[20:22],retained[20:22]);self.assertEqual(retained[20:22],[0,0])
                self.assertEqual(numeric_state(g,pe)[5:14],[0]*9)
                self.assertEqual(final_numeric_state(g,pe)[7],g)
                self.assertEqual(final_numeric_state(g,pe,released=True)[13],1)
                if pe!=2:
                    self.assertEqual(weight_snapshot(g,pe,reset=True)[12],2*g-1)
                    self.assertEqual(weight_snapshot(g,pe)[12],2*g)
                    self.assertEqual(tile_snapshot(g,pe,0)[12:14],[identity(g)[1],g])
            self.assertEqual(mlp_state(g,2,incoming=3,computed=True)[11],g)
            self.assertEqual(mlp_state(g,3,incoming=4,computed=True,tiles=2)[7:12],[128,2,128,128,g])

    def test_receiver_join_orders_and_PE2_preserved_edge_counts(self):
        for g in (1,2,3):
            for edge in (0,1,2):
                pe=2 if edge<2 else 3
                for first,before in ((False,False),(True,False),(True,True)):
                    a=completed_handoff_state(g,edge,pe,receipt_first=first,command_before_ack=before)
                    self.assertEqual(a[23],a[16]);self.assertGreater(a[23],max(a[18],a[21]))
                    self.assertEqual(a[30],int(a[19]<a[18]));self.assertEqual(a[7:11],[1,1,0,0])
            sender=completed_handoff_state(g,2,2)
            self.assertEqual((sender[5],sender[15],sender[16]),(3,2,16))
            self.assertEqual(sender[7:11],[0,0,1,1])

if __name__=='__main__':unittest.main()
