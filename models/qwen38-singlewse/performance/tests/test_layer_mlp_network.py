"""Structural and bounded protocol checks; these are not neural/device results."""
import copy
import json
from collections import deque
from pathlib import Path
import random
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.layer_mlp_network import build_mlp_network,worker_profiles,audit_mlp_network,input_selectors,emit_routes


class MlpNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evidence=Path(__file__).resolve().parents[1]/'evidence'
        plan=json.loads((evidence/'layer-native-schedule-002/layer-schedule.json').read_text())
        cls.cases=[]
        for name,shape in [('layer_00','78x67'),('layer_03','56x94')]:
            stage=next(s for s in plan['stages'] if s['id']==name)
            routes=json.loads((evidence/f'layer-fused-routes-003/routes-{shape}.json').read_text())
            network=build_mlp_network(stage,routes)
            cls.cases.append((stage,routes,network))

    def test_original_native_owners_and_all_compiled_ports(self):
        for stage,routes,network in self.cases:
            profiles=worker_profiles(stage,routes,network)
            self.assertEqual(len(profiles),network['audit']['native_workers'])
            self.assertEqual(sum(w['parameters'].get('fusion_actor',False) for w in profiles),33)
            self.assertEqual(sum(w['parameters'].get('input_quantizer',False) for w in profiles),40)
            self.assertEqual(sum(w['parameters'].get('output_sink',False) for w in profiles),10)
            self.assertTrue(all(w['parameters']['copy_transport'] and w['parameters']['tagged_input'] for w in profiles))
            self.assertFalse(network['executed']);self.assertFalse(network['physical'])
            # Bank changes cannot hide inside new routing/transport parameters.
            for w in profiles:
                self.assertEqual(w['parameters']['bank_words']*4,w['original_matrix_bytes']+w['live_auxiliary_bytes'])
            self.assertLess(len(emit_routes(network)),500000)

    def test_wrong_input_filter_unsupported_swap_and_ungranted_merge_rejected(self):
        stage,_,network=self.cases[0]
        for mutation in ('filter','swap','merge'):
            bad=copy.deepcopy(network)
            r=next(r for r in bad['routes'] if r['color']==2 and 'filter_tag' in r)
            if mutation=='filter':r['filter_tag']+=10000
            elif mutation=='swap':r['color_swap_x']=True
            else:r['rx'].append('RAMP')
            with self.assertRaises(ValueError):audit_mlp_network(stage,bad)

    def test_tagged_frames_preserve_bits_and_ragged_native_parts(self):
        # Distinct arbitrary16-bit values and32-bit scales, reverse/shuffled group
        # arrival. Match the hardware filter's whole-frame upper-half selector.
        rng=random.Random(389);epoch=0x89abcdef
        for stage,_,_ in self.cases:
            for region in stage['regions']:
                if region['role'] not in ('gate_up','down'):continue
                mapping=input_selectors(region);cols=region['matrices'][0]['tile_shape'][1]
                frames=[]
                for f in mapping['frames']:
                    values=[(f['k']*97+i*13)&65535 for i in range(cols)];scale=(f['k']*11379+0x37231122)&0xffffffff
                    halves=[epoch&65535,epoch>>16,f['part'],*values,scale&65535,scale>>16]
                    frame=[f['tag']<<16|v for v in halves]
                    # In-place lower-half compaction used by the native receiver.
                    arena=[half for word in frame for half in [word&65535,word>>16]]
                    for i in range(cols):arena[4+i]=arena[6+2*i]
                    self.assertEqual(arena[4:4+cols],values)
                    self.assertEqual((frame[-2]&65535)|((frame[-1]&65535)<<16),scale)
                    frames.append((f['tag'],f['part'],f['k']))
                rng.shuffle(frames)
                for b in mapping['bindings']:
                    observed=sorted((part,k) for tag,part,k in frames if tag==b['tag'])
                    self.assertEqual(observed,list(enumerate(b['slices'])))

    def test_one_granted_frame_may_reuse_source_before_complete_delivery(self):
        # Model the actual line topology with small per-hop FIFOs and delayed
        # receiving. Enforcing complete-frame receipt before regrant makes every
        # injection globally exclusive, even when prior local DMA has completed.
        _,_,network=self.cases[0]
        sources={s['id']:tuple(s['pe']) for s in network['senders']}
        routes={tuple(r['pe']):r for r in network['routes'] if r['color']==18}
        delta={'EAST':(1,0),'WEST':(-1,0),'SOUTH':(0,1),'NORTH':(0,-1)}
        first=next(pe for pe,r in routes.items() if r['rx']==['RAMP'])
        line=[];pe=first
        while True:
            line.append(pe);direction=routes[pe]['tx'][0]
            if direction=='RAMP':break
            dx,dy=delta[direction];pe=(pe[0]+dx,pe[1]+dy)
        positions={p:i for i,p in enumerate(line)};queues=[deque() for _ in line]
        rng=random.Random(20260928);local_before_remote=False
        # Real186-grant schedule, real frame lengths, all distinct word identities.
        for sequence,g in enumerate(network['grant_schedule']):
            at=positions[sources[g['target']]];expected=[(sequence,i) for i in range(g['words'])]
            arena=expected[:];sent=0;received=[]
            for step in range(100000):
                if sent<len(expected) and len(queues[at])<2 and rng.randrange(4):
                    queues[at].append(arena[sent]);sent+=1
                    if sent==len(expected):
                        local_before_remote |= len(received)<len(expected);arena[:]=[None]*len(arena)
                for i in reversed(range(len(line)-1)):
                    if queues[i] and len(queues[i+1])<2 and rng.randrange(4):queues[i+1].append(queues[i].popleft())
                if queues[-1] and rng.randrange(3):received.append(queues[-1].popleft())
                if len(received)==len(expected):break
            self.assertEqual(received,expected)
            self.assertFalse(any(queues),'Next source cannot inject while an old frame remains')
        self.assertTrue(local_before_remote)

    def test_single_rx_restore_must_wait_for_output_queue_empty(self):
        # A local-complete callback can run with two uncopied router wavelets.
        # Restoring transit then would strand them; qflush delays that change.
        oq=deque([21,22]);rx='RAMP';delivered=[];local_complete=True
        self.assertTrue(local_complete);self.assertTrue(oq)
        with self.assertRaises(AssertionError):
            assert not oq, 'An empty private DMA arena is not an empty OQ'
        while oq:
            self.assertEqual(rx,'RAMP');delivered.append(oq.popleft())
        rx='WEST';self.assertEqual(delivered,[21,22]);self.assertEqual(rx,'WEST')
        # Every generated initial router config has exactly one receive input.
        # Only granted source PEs carry the explicit runtime-injection capability.
        for _,_,network in self.cases:
            sources={tuple(s['pe']) for s in network['senders']}
            for r in network['routes']:
                self.assertEqual(len(r['rx']),1)
                if r.get('grant_selected_injection'):self.assertIn(tuple(r['pe']),sources)

    def test_group_round_robin_does_not_reorder_a_fusion_actors_groups(self):
        for _,routes,network in self.cases:
            by_producer={s['fusion_producer']:s['id'] for s in network['senders'] if s.get('fusion_producer') is not None}
            for actor in routes['actors']:
                actual=[g['index'] for g in network['grant_schedule'] if g['kind']==1 and g['target']==by_producer[actor['producer']]]
                self.assertEqual(actual,actor['groups'])
            output=[row for g in network['grant_schedule'] if g['kind']==2 for row in range(g['first_row'],g['first_row']+g['rows'])]
            self.assertEqual(output,list(range(5120)))


if __name__=='__main__':unittest.main()
