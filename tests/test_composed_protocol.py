import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.composed_protocol import validate_state,validate_ledger


class ComposedProtocolTests(unittest.TestCase):
    def test_ack_before_root_commit_and_consumer_prepare(self):
        states=[[1,2,2,1,4,0,3,2,2,1,265,1],[1,2,2,1,4,0,1,2,2,1,131,1],[1,2,2,1,4,0,1,1,2,1,131,1]]
        events=[[11,1,2,3,4,5,6,7,10,11,8,9,0],[8,1,2,3,0,4,5,6,7,8,0,0,0],[6,2,3,4,0,0,0,0,5,6,1,0,0]]
        validate_ledger();self.assertTrue(validate_state(states,events,1,2,2,1))
        bad=copy.deepcopy(events);bad[0][8],bad[0][11]=bad[0][11],bad[0][8]
        self.assertFalse(validate_state(states,bad,1,2,2,1))
        bad=copy.deepcopy(events);bad[2][10]=0
        self.assertFalse(validate_state(states,bad,1,2,2,1))
        bad=copy.deepcopy(states);bad[2][9]=2
        self.assertFalse(validate_state(bad,events,1,2,2,1))


if __name__=='__main__':unittest.main()
