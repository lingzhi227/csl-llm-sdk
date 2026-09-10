import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StreamPrepareTests(unittest.TestCase):
    def test_full_slab_tail_and_synthetic_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            slab = load('make_wp03_fixture').create(path / 'slab')
            image = path / 'sdk.sif'
            image.touch()
            run = load('prepare_wp03').prepare(ROOT, path / 'run', image, slab, 5120)
            profile = json.loads((run / 'profile.json').read_text())
            self.assertEqual(len(profile['tiles']), 46)
            self.assertEqual(profile['tiles'][-1]['valid'], 80)
            self.assertEqual(json.loads((run / 'slab-source.json').read_text())['kind'],
                             'synthetic_reproduction_fixture')
            raw = (slab / 'slab-row-major.bf16').read_bytes()
            tail = (run / profile['tiles'][-1]['file']).read_bytes()
            for row in (0, 63, 127):
                for column in (0, 79):
                    start = (column * 128 + row) * 2
                    original = (row * 5120 + 5040 + column) * 2
                    self.assertEqual(tail[start:start + 2], raw[original:original + 2])
            self.assertEqual(tail[80 * 128 * 2:], b'\x80\x3f' * (32 * 128))
            manifest = json.loads((run / 'source-manifest.json').read_text())['files']
            for name, digest in manifest.items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), digest)
            self.assertFalse((run / 'out').exists())
            with self.assertRaises(FileExistsError):
                load('make_wp03_fixture').create(slab)
