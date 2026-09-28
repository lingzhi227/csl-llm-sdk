"""Preserve actual consumer banks and reject unsupported complete-stage claims."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from spatial.frontend_resources import audit_frontend_resources
from spatial.mixer_frontend_composition import compose_frontend


class FrontendResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=ROOT/'evidence/layer-mlp-compile-021/source'
        cls.compiled=ROOT/'evidence/layer-backend-compile-023'
        source=cls.compiled/'source'
        profiles=[dict(pe=pe,source=c['source'],parameters=dict(c['parameters']))
                  for c in json.loads((cls.base/'profiles.json').read_text())['profiles'] for pe in c['pes']]
        cls.args=[json.loads((source/'frontend-network.json').read_text()),profiles,
                  json.loads((cls.base/'dialogue-bank-placement.json').read_text()),
                  json.loads((ROOT/'evidence/dialogue-bank-census-001/result.json').read_text()),
                  json.loads((source/'profiles.json').read_text())['samples'],json.loads((cls.compiled/'sram.json').read_text())]

    def test_actual_consumer_admission_does_not_admit_whole_stage(self):
        result=audit_frontend_resources(*self.args)
        self.assertEqual(result['selected_samples'],33)
        self.assertTrue(result['all16_frontend_consumers_keep_complete_original_banks'])
        self.assertEqual((result['actual_max_consumer_bytes_with_stack'],result['actual_min_consumer_margin']),(47984,144))
        self.assertFalse(result['fixed_bank_candidate_accepted'])
        self.assertFalse(result['bank_relocation_only_candidate_accepted'])
        self.assertEqual(result['estimated_matrix_prefix_conflicts'],1275)
        self.assertFalse(result['whole_stage_compiled'])
        self.assertFalse(result['complete_model_speed'])

    def test_missing_elf_changed_bank_and_changed_original_program_are_rejected(self):
        for change in ('elf','bank','stack','parameters'):
            args=deepcopy(self.args)
            if change=='elf':args[5]['records'].pop()
            if change=='bank':next(s for s in args[4] if 'frontend_group' in s['parameters'])['parameters']['bank_words']-=32
            if change=='stack':args[5]['records'][0]['stack_allowance_bytes']=2048
            if change=='parameters':args[4][0]['parameters']['mixer_rank_parity']=1
            with self.subTest(change=change),self.assertRaises(ValueError):audit_frontend_resources(*args)

    def test_current_fused_programs_match_real_compiled_programs(self):
        files={p.name:p.read_bytes() for p in self.base.glob('*.csl')}
        original=deepcopy(files);profiles=deepcopy(self.args[1])
        region=next(r for r in json.loads((self.base/'joint-stage.json').read_text())['regions'] if r['role']=='mix')
        setups=json.loads((self.base/'mixer-setups.json').read_text())
        compose_frontend(files,profiles,region,setups)
        for name,raw in original.items():self.assertEqual(files[name],raw,name)
        for name,raw in files.items():
            if name.endswith('.csl') and name not in original:
                self.assertEqual(raw,(self.compiled/'source'/name).read_bytes(),name)
        self.assertEqual(sum(p['source'].startswith('frontend_') for p in profiles),1301)


if __name__=='__main__':unittest.main()
