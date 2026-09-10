import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
spec=importlib.util.spec_from_file_location('prepare_wp01',ROOT/'tools/prepare_wp01.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from qwen38.numerics import weights_from_bits


class PrepareGEMVTests(unittest.TestCase):
    def test_synthetic_fixture_transpose_and_frozen_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);image=path/'sdk.sif';image.touch()
            run=module.prepare(ROOT,path/'run',image)
            raw=(run/'tile-row-major.bf16').read_bytes();packed=(run/'tile-column-major.bf16').read_bytes()
            restored=b''.join(packed[(c*128+r)*2:(c*128+r)*2+2] for r in range(128) for c in range(112))
            self.assertEqual(restored,raw)
            self.assertEqual(len(weights_from_bits(raw)),14336)
            self.assertEqual(json.loads((run/'tile-source.json').read_text())['kind'],'synthetic_reproduction_fixture')
            manifest=json.loads((run/'source-manifest.json').read_text())['files']
            self.assertIn('numerical-policy.json',manifest)
            for name,digest in manifest.items():
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),digest)
            self.assertFalse((run/'out').exists())
