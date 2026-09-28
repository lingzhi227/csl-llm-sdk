"""Protect complete bank addresses and leases during production integration."""
from collections import Counter
import json
from pathlib import Path
import unittest

from tools.build_resident_gdn_stage import build, retained_routes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'evidence/layer-mlp-compile-027/source'


class ResidentGdnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files, cls.width, cls.height, cls.profiles = build()

    def test_all_original_banks_and_routes_are_identical(self):
        for name in ('gdn-bank-placement.json', 'worker-setups.json', 'mixer-setups.json',
                     'gdn-mlp-network.json', 'mixer-projections.json',
                     'frontend-network.json', 'device-network.json'):
            self.assertEqual(self.files[name], (BASE/name).read_bytes(), name)
        old = (BASE/'layout.csl').read_bytes()
        new = self.files['layout.csl']
        self.assertEqual(old[old.index(b' const route_0='):], new[new.index(b' const route_0='):])
        before = {tuple(pe):c['parameters'] for c in json.loads((BASE/'profiles.json').read_text())['profiles'] for pe in c['pes']}
        self.assertEqual(len(before), self.width*self.height)
        for p in self.profiles:
            for key in ('bank_words', 'gdn_state_word', 'gdn_columns', 'gdn_head', 'gdn_first'):
                self.assertEqual(p['parameters'].get(key), before[tuple(p['pe'])].get(key))

    def test_endpoint_colors_do_not_claim_an_existing_route(self):
        occupied = {(*r['pe'],r['color']) for r in retained_routes(self.files)}
        seen = Counter()
        for p in self.profiles:
            params = p['parameters']
            if 'gdn_columns' in params:
                colors = params['gdn_input_color'],params['gdn_output_color']
                self.assertIn(colors[1], [0,1,2,3,4,5,6,7,8,9,12,13,16,17,20])
                seen['worker'] += 1
            elif 'frontend_group' in params:
                colors = params['gdn_send_color'],params['gdn_return_color']
                seen['frontend'] += 1
            else:
                continue
            for color in colors:
                self.assertNotIn((*p['pe'],color), occupied)
        self.assertEqual(seen, {'worker':753,'frontend':16})

    def test_packet_callback_cannot_retire_successor_output(self):
        raw = self.files['resident_frontend_native_001_device_mixer_layer_mlp_standby.csl']
        sent = raw.split(b'task gdn_packet_sent() void {',1)[1].split(b'\n}',1)[0]
        self.assertIn(b'frontend.packet_consumed', sent)
        self.assertNotIn(b'output_consumed', sent)
        # One explicit successor-consumption port remains. Completion merely
        # notifies/snapshots, so projected output cannot be overwritten early.
        self.assertEqual(raw.count(b'frontend.output_consumed('), 1)
        self.assertIn(b'task frontend_output_ready() void {frontend_snapshot();}', raw)
        self.assertNotIn(b'else if(c[1]==9)', raw)
        self.assertNotIn(b'else if(c[1]==10)', raw)
        self.assertIn(b'@assert(gdn_graph_drained());gdn_next_head=0;', raw)
        self.assertIn(b'gdn_markers==gdn_worker_count', raw)
        self.assertIn(b'gdn_core_frames==192', raw)
        self.assertNotIn(b'captured', raw)

    def test_new_ports_are_explicitly_unrouted(self):
        proof = json.loads(self.files['resident-gdn-composition.json'])
        self.assertFalse(proof['new_ports_routed'])
        self.assertFalse(proof['executed'])
        self.assertFalse(proof['physical'])
        self.assertFalse(proof['full_layer'])
        self.assertEqual(proof['original_bank_bytes'], 388094976)

    def test_refinement_preserves_every_original_state_coordinate(self):
        from spatial.gdn_placement import GdnBankPlacement, STATE_FIRST
        files,_,_,profiles=build((0,1,2),True)
        before=json.loads(self.files['gdn-bank-placement.json'])
        after=json.loads(files['gdn-bank-placement.json'])
        self.assertEqual(after['stage'],before['stage'])
        self.assertEqual(after['bank_prefixes'],before['bank_prefixes'])
        self.assertEqual(after['allocated_bytes'],before['allocated_bytes'])
        self.assertEqual(len(after['workers']),789)
        seen=bytearray(786432)
        for w in after['workers']:
            self.assertEqual(w['columns']%2,0)
            for key in range(128):
                for value in range(w['first'],w['first']+w['columns']):
                    canonical=(w['head']*128+key)*128+value
                    self.assertFalse(seen[canonical]);seen[canonical]=1
        self.assertTrue(all(seen))
        def identities(plan):
            return {(s['namespace'],s['page_start']+i)for s in plan['spans']for i in range(s['pages'])}
        self.assertEqual(identities(before),identities(after))
        self.assertEqual(len(identities(after)),7085)
        resolver=GdnBankPlacement(json.loads(files['frontend-stage.json']),json.loads(files['mlp-bank-placement.json']),after)
        for w in after['workers']:
            for key,value in ((0,w['first']),(127,w['first']+w['columns']-1)):
                word=resolver.state_word(w['head'],key,value)
                self.assertEqual(word,dict(pe=w['pe'],byte_offset=4*(w['state_word']+key*w['columns']+value-w['first']),bytes=4))
                page=STATE_FIRST+w['head']*512+(key//4)*16+value//8
                spans=resolver.recurrent_spans(page)
                self.assertEqual(sum(s['bytes']for s in spans),128)
                self.assertTrue(any(s['pe']==word['pe'] and s['byte_offset']<=word['byte_offset']<s['byte_offset']+s['bytes']for s in spans))
        # No first-three-group route or worker is silently reinterpreted.
        for group in (0,1,2):
            select=lambda p:sorted((w for w in p['workers']if w['head']//3==group),key=lambda w:w['pe'])
            self.assertEqual(select(before),select(after))
        occupied={(*r['pe'],r['color'])for r in retained_routes(files)}
        for p in profiles:
            if 'gdn_columns'in p['parameters']:
                for k in ('gdn_input_color','gdn_output_color'):
                    self.assertNotIn((*p['pe'],p['parameters'][k]),occupied)


if __name__ == '__main__':
    unittest.main()
