import hashlib
import importlib.util
import json
from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'core'))
spec=importlib.util.spec_from_file_location('prepare_wp02',ROOT/'tools/prepare_wp02.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PrepareChainTests(unittest.TestCase):
    def test_chain_ledger_and_policy_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);image=path/'sdk.sif';image.touch()
            run=module.prepare(ROOT,path/'run',image)
            source=json.loads((run/'source-manifest.json').read_text())['files']
            self.assertIn('resource-ledger.json',source)
            self.assertIn('numerical-policy.json',source)
            for name,digest in source.items():self.assertEqual(hashlib.sha256((run/name).read_bytes()).hexdigest(),digest)
            steps=json.loads((run/'steps.json').read_text())
            self.assertIn('--fabric-dims=9,3',steps['steps'][0]['argv'])
            self.assertFalse((run/'out').exists())
