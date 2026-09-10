import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CpuProfileTests(unittest.TestCase):
    def test_long_deadline_rejected_before_any_service_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            work = cache/'run'
            work.mkdir()
            spec = work/'steps.json'
            spec.write_text(json.dumps({'planned_write_bytes': 1, 'steps': [
                {'name': 'must_not_launch', 'seconds': 61, 'argv': ['missing-command']}]}))
            proc = subprocess.run([sys.executable, str(ROOT/'tools/guarded_run.py'),
                                   '--cache', str(cache), '--work', str(work), '--spec', str(spec),
                                   '--profile', 'cpu-reference'], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('1..60 seconds', proc.stderr)
            self.assertFalse((work/'supervisor.json').exists())

    def test_reference_bundle_cannot_fall_back_to_larger_default_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            work = cache/'run'
            work.mkdir()
            spec = work/'steps.json'
            spec.write_text(json.dumps({'required_profile': 'cpu-reference', 'planned_write_bytes': 1,
                                        'steps': [{'name': 'no_launch', 'seconds': 1, 'argv': ['missing-command']}]}))
            proc = subprocess.run([sys.executable, str(ROOT/'tools/guarded_run.py'), '--cache', str(cache),
                                   '--work', str(work), '--spec', str(spec)], capture_output=True, text=True, timeout=5)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('profile differs', proc.stderr)
            self.assertFalse((work/'supervisor.json').exists())
