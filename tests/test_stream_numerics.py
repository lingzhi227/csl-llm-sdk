import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.stream_numerics import check, reference, vectors


class StreamNumericalTests(unittest.TestCase):
    def test_reference_detects_lost_prior_tiles_and_wrong_global_onehot(self):
        weights = [0.0] * (128 * 5120)
        for row in range(128):
            weights[row * 5120] = 0.5
            weights[row * 5120 + 5119] = 0.25
        vector = [1.0] * 5120
        expected, bounds = reference(weights, vector, 5120)
        self.assertEqual(expected, [0.75] * 128)
        self.assertFalse(check([0.25] * 128, expected, bounds)['passed'])
        onehot = vectors(5120)[2][1]
        expected, bounds = reference(weights, onehot, 5120)
        self.assertTrue(check([0.25] * 128, expected, bounds, exact=True)['passed'])
        self.assertFalse(check([0.0] * 128, expected, bounds, exact=True)['passed'])
        prefix, prefix_bounds = reference(weights, onehot, 5040)
        self.assertTrue(check([0.0] * 128, prefix, prefix_bounds, exact=True)['passed'])
