"""Retain a complete conversation, both work slots and all original weights."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial.dialogue_placement import lower_dialogue_banks, audit_dialogue_banks, DialogueAuxiliaryPlacement
from spatial.joint_placement import JointAuxiliaryPlacement


class DialoguePlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e = ROOT/'evidence'
        stage = next(s for s in json.loads((e/'layer-native-schedule-002/layer-schedule.json').read_text())['stages'] if s['id'] == 'layer_00')
        cls.original = next(r for r in stage['regions'] if r['role'] == 'mix')
        source = e/'layer-mlp-compile-018/source'
        cls.region = next(r for r in json.loads((source/'joint-stage.json').read_text())['regions'] if r['role'] == 'mix')
        cls.before = json.loads((source/'joint-bank-placement.json').read_text())
        cls.profiles = [dict(pe=pe, source=c['source'], parameters=dict(c['parameters']))
                        for c in json.loads((source/'profiles.json').read_text())['profiles'] for pe in c['pes']]
        cls.plan = lower_dialogue_banks(cls.original, cls.region, cls.before, cls.profiles)
        cls.resolver = DialogueAuxiliaryPlacement(cls.original, cls.region, cls.before, cls.plan)
        cls.source = JointAuxiliaryPlacement(cls.original, cls.region, cls.before)

    def test_complete_state_weights_and_two_work_slots_survive(self):
        plan = self.plan
        self.assertEqual((plan['retained_pages'], plan['removed_state_pages']), (28605, 25056))
        self.assertEqual((plan['source_allocated_bytes'], plan['allocated_bytes'], plan['reclaimed_bytes']),
                         (396953216, 393746048, 3207168))
        self.assertEqual(sum(p['parameters'].get('bank_words', 0)*4 for p in self.profiles), plan['allocated_bytes'])
        addresses = set()
        for obj in self.original['auxiliary']:
            count = obj['bytes_per_request']//128 if obj['kind'] == 'request_state' else obj['pages']
            for page in range(obj['page_start'], obj['page_start']+count):
                owner = self.resolver.owner(page)
                self.assertEqual(owner['pe'], self.source.owner(page)['pe'])
                address = (*owner['pe'], owner['byte_offset'])
                self.assertNotIn(address, addresses)
                addresses.add(address)
            for page in range(obj['page_start']+count, obj['page_start']+obj['pages']):
                with self.assertRaises(ValueError): self.resolver.owner(page)
        self.assertEqual(len(addresses), 28605)

    def test_all_48_recurrent_heads_resolve_without_alias(self):
        addresses = set()
        for head in range(48):
            for key in range(32):
                for value in range(16):
                    a = self.resolver.gdn_owner(0, head, key, value)
                    addresses.add((*a['pe'], a['byte_offset']))
        self.assertEqual(len(addresses), 24576)
        for coordinate in [(1, 0, 0, 0), (0, 48, 0, 0), (0, 0, 32, 0), (0, 0, 0, 16)]:
            with self.assertRaises(ValueError): self.resolver.gdn_owner(*coordinate)

    def test_corrupt_maps_cannot_drop_live_state_or_work_slots(self):
        for change in ('missing', 'alias', 'wrong_owner', 'state_count', 'work_slots', 'total', 'prefix', 'source_request'):
            p = deepcopy(self.plan)
            if change == 'missing': p['spans'].pop(0)
            if change == 'alias': p['spans'][1].update(pe=p['spans'][0]['pe'], byte_offset=p['spans'][0]['byte_offset'])
            if change == 'wrong_owner': p['spans'][0]['pe'] = p['spans'][-1]['pe']
            if change == 'state_count': p['objects'][5]['retained_pages'] -= 1
            if change == 'work_slots': p['work_slots'] = 1
            if change == 'total': p['reclaimed_bytes'] += 128
            if change == 'prefix': p['bank_prefixes'][0]['bytes'] -= 128
            if change == 'source_request': p['retained_source_request'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit_dialogue_banks(self.original, self.region, self.before, p)

    def test_profile_mismatch_is_rejected_before_mutation(self):
        profiles = deepcopy(self.profiles)
        snapshot = deepcopy(profiles)
        with self.assertRaisesRegex(ValueError, 'qualified source banks'):
            lower_dialogue_banks(self.original, self.region, self.before, profiles)
        self.assertEqual(profiles, snapshot)

    def test_complete_builder_preserves_neural_sources_routes_and_matrix_descriptors(self):
        sys.path.insert(0, str(ROOT/'tools'))
        from stage_layer_mlp_compile import build
        files, width, height, profiles = build('layer_00', shared_inputs=True, norm_bridge=True,
                mixer_projections=True, joint_banks=True, routed_control=True, dialogue=True)
        self.assertEqual((width, height, len(profiles)), (78, 146, 11388))
        source = ROOT/'evidence/layer-mlp-compile-018/source'
        for path in source.glob('*.csl'):
            if path.name != 'layout.csl':
                self.assertEqual(files[path.name], path.read_bytes(), path.name)
        for name in ['mixer-setups.json', 'worker-setups.json', 'device-network.json', 'network-binding.json',
                     'joint-stage.json', 'joint-bank-placement.json']:
            self.assertEqual(files[name], (source/name).read_bytes(), name)
        self.assertEqual(files['layout.csl'].split(b' const route_0=')[1],
                         (source/'layout.csl').read_bytes().split(b' const route_0=')[1])
        old = {tuple(pe): (p['source'], p['parameters'])
               for p in json.loads((source/'profiles.json').read_text())['profiles'] for pe in p['pes']}
        for p in profiles:
            name, parameters = old[tuple(p['pe'])]
            self.assertEqual(p['source'], name)
            self.assertEqual({k: v for k, v in p['parameters'].items() if k != 'bank_words'},
                             {k: v for k, v in parameters.items() if k != 'bank_words'})
            self.assertLessEqual(p['parameters'].get('bank_words', 0), parameters.get('bank_words', 0))
        self.assertEqual(json.loads(files['profiles.json'])['bank_specialization'], 'dialogue-bank-placement.json')


if __name__ == '__main__': unittest.main()
