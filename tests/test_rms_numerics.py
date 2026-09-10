import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from qwen38.rms_numerics import bf16_rne,bf16_of_real,from_bits,fixtures,reference,close,N


class RmsTests(unittest.TestCase):
    def test_signed_midpoints_even_odd_adjacent_and_zero(self):
        cases={0x3f808000:0x3f80,0x3f818000:0x3f82,0xbf808000:0xbf80,0xbf818000:0xbf82,
               0x3f807fff:0x3f80,0x3f808001:0x3f81,0xbf807fff:0xbf80,0xbf808001:0xbf81,
               0:0,0x80000000:0x8000,0x3fff8000:0x4000,0xbfff8000:0xc000}
        for raw,expected in cases.items():self.assertEqual(bf16_rne(raw),expected)
        with self.assertRaises(ValueError):bf16_rne(0x7f800000)

    def test_fp64_oracle_avoids_double_rounding(self):
        midpoint=from_bits(0x3f808000)
        self.assertEqual(bf16_of_real(midpoint+2**-30),0x3f81)
        self.assertEqual(bf16_rne(0x3f808000),0x3f80)

    def test_offset_gain_and_wrong_zeroing_rejected(self):
        _,x,w=list(fixtures())[1];ref=reference(x,w)
        self.assertFalse(close([0.0]*N,ref['expected'],ref['bounds'])['passed'])
        ref=reference(x,[-1.0]*N)
        self.assertEqual(ref['expected'],[0.0]*N)

    def test_preparer_freezes_rounding_adversary_and_sram_gate(self):
        spec=importlib.util.spec_from_file_location('p',ROOT/'tools/prepare_wp05.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);image=path/'sdk.sif';image.touch()
            run=module.prepare(ROOT,path/'run',image)
            self.assertGreater(json.loads((run/'conversion-fixtures.json').read_text())['gain_before_cast_fixture_differences'],0)
            self.assertEqual(json.loads((run/'sram-policy.json').read_text())['stack_allowance_bytes'],4096)
            self.assertEqual(json.loads((run/'steps.json').read_text())['steps'][1]['seconds'],180)
