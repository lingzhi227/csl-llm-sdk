import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.attention_numerics import D, append, fixtures, reference, exact_sum_products


class AttentionNumericsTests(unittest.TestCase):
    def test_current_value_and_invalid_poison_exclusion(self):
        data = fixtures()
        token = data['tokens'][0]
        cache = append(data['initial_cache'], token)
        result = reference(token, cache)
        self.assertEqual(result['output'], [0.5] + [0.0] * (D - 1))
        self.assertEqual(result['probability_bf16'], [1.0])
        self.assertEqual(cache[D:8*D], data['initial_cache'][D:8*D])
        self.assertEqual(data['initial_cache'][0], -1/16)

    def test_negative_scores_reset_and_overflow(self):
        data = fixtures()
        cache = data['initial_cache']
        for token in data['tokens'][:8]:
            cache = append(cache, token)
            if token['token'] == 4:
                self.assertLess(max(reference(token, cache)['scaled_bf16']), 0)
        snapshot = cache.copy()
        with self.assertRaises(ValueError):
            append(cache, {**data['tokens'][7], 'token': 9})
        self.assertEqual(cache, snapshot)
        token = data['tokens'][8]
        reset_cache = append(cache, token)
        self.assertEqual(reset_cache[D:8*D], cache[D:8*D])
        self.assertEqual(reference(token, reset_cache)['output'], [0.0]*D)

    def test_exact_reduction_proof_rejects_lost_small_term(self):
        self.assertTrue(exact_sum_products([0.5]*D, [-0.125]*D))
        self.assertFalse(exact_sum_products([1.0, 2**-30], [1.0, 1.0]))


if __name__ == '__main__':
    unittest.main()
