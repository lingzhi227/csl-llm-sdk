from pathlib import Path
from dataclasses import replace
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.banks import BankPolicy,local_count,tile_owner

class BankTests(unittest.TestCase):
    def test_prefix_count_matches_independent_enumeration(self):
        for n in (1,2,7,17):
            for phase in (0,n//2,n-1):
                for length in (0,1,n-1,n,n+1,3*n+5):
                    counts=[0]*n
                    for tile in range(length):counts[(phase+tile)%n]+=1
                    self.assertEqual([local_count(length,pe,n,phase) for pe in range(n)],counts)
    def test_local_offsets_do_not_overlap_across_matrices(self):
        n=7;phase=3;start=0;occupied=set()
        for length in (11,2,28,9):
            for tile in range(length):
                pe=tile_owner(start,tile,n,phase)
                slot=local_count(start+tile,pe,n,phase)
                self.assertNotIn((pe,slot),occupied);occupied.add((pe,slot))
            start+=length
        self.assertEqual(len(occupied),start)
    def test_cannot_silently_raise_memory_or_change_scale_granularity(self):
        for p in (replace(BankPolicy(),sram_ceiling=65536),replace(BankPolicy(),rows_per_tile=3),replace(BankPolicy(),columns_per_tile=64),replace(BankPolicy(),reserved_actor_pes=870000)):
            with self.assertRaises(ValueError):p.validate()
