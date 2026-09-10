from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.stream_schedule import Schedule,Tile,tiles


class StreamTests(unittest.TestCase):
    def test_full_extent_and_tail(self):
        plan=tiles(5120,1);self.assertEqual(len(plan),46)
        self.assertEqual(plan[-1],Tile(1,45,5040,80))
        s=Schedule(5120,1)
        for tile in plan:s.submit(tile)
        self.assertEqual(s.finalize(),1)

    def test_missing_duplicate_order_generation_and_tail(self):
        s=Schedule(304,1)
        with self.assertRaises(ValueError):s.finalize()
        for bad in (Tile(1,1,112,112),Tile(2,0,0,112),Tile(1,0,0,80)):
            with self.assertRaises(ValueError):s.submit(bad)
        s.submit(Tile(1,0,0,112))
        with self.assertRaises(ValueError):s.submit(Tile(1,0,0,112))
        s.submit(Tile(1,1,112,112))
        with self.assertRaises(ValueError):s.submit(Tile(1,2,224,112))
        s.submit(Tile(1,2,224,80));s.finalize()
        with self.assertRaises(ValueError):s.finalize()
        with self.assertRaises(ValueError):s.submit(Tile(1,3,304,1))
