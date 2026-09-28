"""Bind the reproducible emitted programs to the successful complete compiler."""
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_frontend_stage import build


class FrontendBuilderTests(unittest.TestCase):
    def test_full_source_and_active_banks_match_admitted_snapshot(self):
        files,width,height,profiles=build()
        self.assertEqual((width,height,len(profiles)),(78,146,11388))
        source=ROOT/'evidence/layer-mlp-compile-022/source'
        for name,data in files.items():
            if name.endswith('.csl') or name in ('profiles.json','mixer-setups.json','frontend-bank-placement.json','frontend-placement.json','compact-mixer.json'):
                self.assertEqual(data,(source/name).read_bytes(),name)
        actual=json.loads((source.parent/'sram-summary.json').read_text())
        self.assertTrue(actual['passed'])
        self.assertEqual((actual['application_pes'],actual['compiled_elf_images']),(11388,7919))


if __name__=='__main__':unittest.main()
