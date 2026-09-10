"""Exact new SDK admission boundaries; no systemd/container/SDK calls."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.wp16_sdk_admission import CANDIDATE, PROFILE, STEPS, GENERATED, application_elf_names, compare_compiled_states, validate


class SDKAdmissionTests(unittest.TestCase):
    def test_runtime_generated_allowlist_does_not_relax_original_identity(self):
        frozen=dict(files={'out/bin/out_0_0.elf':dict(bytes=1,sha256='synthetic')},
                    compiled_bytes=1,footprints={},pes={},passed=True)
        actual=copy.deepcopy(frozen)
        actual['files'].update(copy.deepcopy(GENERATED));actual['compiled_bytes']+=54952
        self.assertEqual(compare_compiled_states(actual,frozen,after_runtime=True),GENERATED)
        with self.assertRaisesRegex(ValueError,'before simulator'):compare_compiled_states(actual,frozen)
        for name in ('out/bin/out_0_0.elf','out/generated/coord.elf'):
            bad=copy.deepcopy(actual);bad['files'][name]['sha256']='changed'
            with self.assertRaises(ValueError):compare_compiled_states(bad,frozen,after_runtime=True)
        bad=copy.deepcopy(actual);bad['files']['out/generated/unexpected.elf']=dict(bytes=1,sha256='other')
        with self.assertRaisesRegex(ValueError,'auxiliary'):compare_compiled_states(bad,frozen,after_runtime=True)

    def test_edge_ELFs_stay_distinct_from_exact_application_PE_set(self):
        application = {'out/bin/out_' + str(pe) + '_0.elf' for pe in range(4)}
        support = {'out/' + side + '/bin/out_' + str(pe) + '_0.elf'
                   for side, count in [('west', 3), ('east', 2)] for pe in range(count)}
        files = application | support | {'out/out.json', 'out/bin/out_rpc.json'}
        self.assertEqual(set(application_elf_names(files)), application)
        self.assertEqual(len(files), 11)
        with self.assertRaises(ValueError):
            application_elf_names(files - {'out/bin/out_1_0.elf'})
        for extra in ('out/bin/out_4_0.elf', 'out/bin/unexpected.elf'):
            with self.assertRaises(ValueError):
                application_elf_names(files | {extra})

    def admission(self):
        return dict(package='WP16', candidate=CANDIDATE, candidate_limit=1,
                    sdk_execution_authorized=True, manifest_sha256='test-only-not-real',
                    profile=copy.deepcopy(PROFILE), steps=copy.deepcopy(STEPS),
                    original_model_rerun_authorized=False, new_weight_preparation_authorized=False)

    def test_scope_resources_and_admission_are_exact(self):
        validate(self.admission(), 'test-only-not-real')
        for name, value in [('cpu_affinity', [0, 1]), ('simulation_seconds', 361),
                            ('swap_bytes', 1), ('input_indices', [0, 1]), ('pes', 5),
                            ('full17408_down_output', True)]:
            altered = self.admission(); altered['profile'][name] = value
            with self.assertRaises(ValueError):
                validate(altered, 'test-only-not-real')
        for name, value in [('candidate_limit', 2), ('sdk_execution_authorized', False),
                            ('manifest_sha256', 'different'), ('new_weight_preparation_authorized', True)]:
            altered = self.admission(); altered[name] = value
            with self.assertRaises(ValueError):
                validate(altered, 'test-only-not-real')

    def test_reorder_or_command_change_cannot_bypass_serial_entry(self):
        for change in ('reverse', 'argv', 'time'):
            altered = self.admission()
            if change == 'reverse':
                altered['steps']['steps'].reverse()
            elif change == 'argv':
                altered['steps']['steps'][1]['argv'] = ['python', 'driver.py']
            else:
                altered['steps']['steps'][1]['seconds'] = 361
            with self.assertRaises(ValueError):
                validate(altered, 'test-only-not-real')


if __name__ == '__main__':
    unittest.main()
