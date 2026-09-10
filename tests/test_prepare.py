import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_wp00',ROOT/'tools/prepare_wp00.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PrepareTests(unittest.TestCase):
    def test_new_bundle_hashes_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory); image=path/'dummy.sif'; image.touch()
            run=module.prepare(ROOT,path/'run',image,'/usr/bin/python3')
            manifest=json.loads((run/'source-manifest.json').read_text())['files']
            for name,digest in manifest.items():
                self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),digest)
            self.assertFalse((run/'out').exists())
            with self.assertRaises(FileExistsError):
                module.prepare(ROOT,run,image,'/usr/bin/python3')
