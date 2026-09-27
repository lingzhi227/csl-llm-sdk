"""Larger measured artifacts cannot broaden unrelated hardware profiles."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('performance_backend',Path(__file__).resolve().parents[1]/'runtime/backend.py')
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)

class ArtifactProfileTests(unittest.TestCase):
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
