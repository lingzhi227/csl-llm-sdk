from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.locking import heavy_lock


class LockTests(unittest.TestCase):
    def test_second_holder_rejected_and_release_preserves_inode(self):
        with tempfile.TemporaryDirectory() as directory, patch('qwen38.locking.identity', return_value={'pid': 123, 'start_ticks':'456', 'host':'fixture'}):
            path = Path(directory)/'heavy.lock'
            with heavy_lock(path):
                inode = path.stat().st_ino
                with self.assertRaisesRegex(RuntimeError, 'Heavy task lock held'):
                    with heavy_lock(path):
                        self.fail('Second owner entered')
            with heavy_lock(path):
                self.assertEqual(inode, path.stat().st_ino)
