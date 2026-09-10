import copy
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.resource_ledger import LEDGER,check


class LedgerTests(unittest.TestCase):
    def test_provider_collision_rejected(self):
        self.assertTrue(check(LEDGER))
        bad=copy.deepcopy(LEDGER);bad['application']['local_tasks'].append(24)
        with self.assertRaises(ValueError):check(bad)
        bad=copy.deepcopy(LEDGER);bad['dsr_leases'].append(bad['dsr_leases'][0])
        with self.assertRaises(ValueError):check(bad)
