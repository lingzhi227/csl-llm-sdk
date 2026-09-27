import importlib.util
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'runtime'))
sys.path.insert(0, str(root / 'performance/runtime'))
spec = importlib.util.spec_from_file_location('performance_sram_coverage', root / 'performance/runtime/check_sram.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class ElfCoverageTests(unittest.TestCase):
    def test_shared_image_is_not_a_missing_pe_and_overlap_is_rejected(self):
        rectangles = {'a': [(4, 1, 1, 1), (5, 1, 1, 1)], 'b': [(4, 2, 2, 1)]}
        self.assertEqual(len(module.cover(rectangles, (2, 2), (4, 1))), 4)
        for invalid in [{'a': [(4, 1, 2, 1)]},
                        {**rectangles, 'extra': [(4, 1, 1, 1)]},
                        {**rectangles, 'extra': [(3, 1, 1, 1)]}]:
            with self.assertRaises(ValueError):
                module.cover(invalid, (2, 2), (4, 1))
