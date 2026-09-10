import sys
from pathlib import Path
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.resources import Budget, GIB, assess, cache_usage, memory_available


class ResourceTests(unittest.TestCase):
    def test_available_includes_reclaimable_memory(self):
        self.assertEqual(memory_available('MemFree: 1 kB\nMemAvailable: 8192 kB\n'), 8388608)

    def test_no_guess_without_available(self):
        with self.assertRaises(ValueError):
            memory_available('MemFree: 100000 kB\n')

    def test_exact_boundary_and_ram_refusal(self):
        b = Budget()
        self.assertTrue(assess(b, 28*GIB, 52*GIB, 0, 0, 20*GIB)['admitted'])
        self.assertFalse(assess(b, 28*GIB-1, 52*GIB, 0, 0, 20*GIB)['admitted'])

    def test_disk_and_cache_reserves(self):
        b = Budget()
        self.assertFalse(assess(b, 30*GIB, 32*GIB, 0, 0, 1)['admitted'])
        self.assertFalse(assess(b, 30*GIB, 90*GIB, 20*GIB, 0, 1)['admitted'])

    def test_boolean_and_negative_rejected(self):
        with self.assertRaises(ValueError):
            Budget(job_ram=True)
        for value in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                assess(Budget(), 30*GIB, 90*GIB, 0, 0, value)

    def test_cache_bounded_and_does_not_follow_links(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'tile').write_bytes(b'abc')
            size, entries = cache_usage(root, 2)
            self.assertGreaterEqual(size, 3)
            self.assertEqual(entries, 1)
            (root/'outside').symlink_to('/tmp')
            with self.assertRaises(ValueError):
                cache_usage(root, 3)
            with self.assertRaises(ValueError):
                cache_usage(root, 1)


if __name__ == '__main__':
    unittest.main()
