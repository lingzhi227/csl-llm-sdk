"""Larger measured artifacts cannot broaden unrelated hardware profiles."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('performance_backend',Path(__file__).resolve().parents[1]/'runtime/backend.py')
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)

class ArtifactProfileTests(unittest.TestCase):
    def test_all_original_gdn_slices_require_exact_bounded_geometry(self):
        profile=dict(original_gdn_columns=True,full_model=False,application=[251,6],application_pes=1506,
                     gdn_shape=[48,128,128],gdn_workers=753,fabric_offset=[4,1],
                     revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',artifact_single_message_limit=16<<20)
        self.assertTrue(backend.gdn_columns_profile(profile))
        self.assertEqual(backend.message_limit(profile),16<<20)
        for key in profile:
            if key=='artifact_single_message_limit':continue
            changed=dict(profile);changed.pop(key)
            self.assertFalse(backend.gdn_columns_profile(changed))
            with self.assertRaises(ValueError):backend.message_limit(changed)
        with self.assertRaises(ValueError):backend.message_limit(dict(profile,artifact_single_message_limit=(16<<20)+1))

    def test_complete_mlp_has_exact_separate_64_mib_gate(self):
        profile=dict(complete_original_mlp=True,stage='layer_00',application=[78,146],application_pes=11388,
                     mlp_shape=[5120,17408,5120],fabric_offset=[67,1],
                     revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',artifact_single_message_limit=64<<20)
        self.assertTrue(backend.complete_mlp_profile(profile))
        self.assertEqual(backend.message_limit(profile),64<<20)
        for key in profile:
            if key=='artifact_single_message_limit':continue
            changed=dict(profile);changed.pop(key)
            self.assertFalse(backend.complete_mlp_profile(changed))
            with self.assertRaises(ValueError):backend.message_limit(changed)
        for value in [0,(64<<20)+1,128<<20]:
            with self.assertRaises(ValueError):backend.message_limit(dict(profile,artifact_single_message_limit=value))

    def test_complete_mixer_requires_original_five_shapes_and_geometry(self):
        profile=dict(complete_original_mixer=True,stage='layer_00',application=[78,146],application_pes=11388,
                     mixer_shapes=[[10240,5120],[6144,5120],[48,5120],[48,5120],[5120,6144]],fabric_offset=[67,1],
                     revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',artifact_single_message_limit=128<<20)
        self.assertTrue(backend.complete_mixer_profile(profile))
        self.assertFalse(backend.complete_mlp_profile(profile))
        self.assertEqual(backend.message_limit(profile),128<<20)
        for value in [0,(128<<20)+1]:
            with self.assertRaises(ValueError):backend.message_limit(dict(profile,artifact_single_message_limit=value))
        for key in profile:
            if key=='artifact_single_message_limit':continue
            changed=dict(profile);changed.pop(key)
            self.assertFalse(backend.complete_mixer_profile(changed))
            with self.assertRaises(ValueError):backend.message_limit(changed)

    def test_larger_archive_requires_every_exact_matrix_geometry_field(self):
        profile=dict(complete_original_matrix=True,application=[631,2],application_pes=1262,
                     matrix_shape=[48,5120],fabric_offset=[123,706],artifact_single_message_limit=16<<20)
        self.assertEqual(backend.message_limit(profile),16<<20)
        for key in ['complete_original_matrix','application','application_pes','matrix_shape','fabric_offset']:
            changed=dict(profile);changed.pop(key)
            with self.assertRaises(ValueError):backend.message_limit(changed)
        for value in [0,(16<<20)+1,128<<20]:
            with self.assertRaises(ValueError):backend.message_limit(dict(profile,artifact_single_message_limit=value))

    def test_small_component_bound_and_existing_full_profile_remain_distinct(self):
        self.assertEqual(backend.message_limit(dict(artifact_single_message_limit=8<<20)),8<<20)
        with self.assertRaises(ValueError):backend.message_limit(dict(artifact_single_message_limit=(8<<20)+1))
        self.assertEqual(backend.message_limit(dict(full_resident=True)),128<<20)
