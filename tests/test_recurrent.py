"""Adversarial recurrence and protocol checks, independent of SDK availability."""
import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.recurrent_numerics import initial_state,raw_tokens,step,check,f32
from qwen38.recurrent_protocol import validate_header,validate_state,validate_ledger


class RecurrentTests(unittest.TestCase):
    def test_frame_identity_and_count(self):
        good=[2,1,3]+[0]*128
        self.assertTrue(validate_header(good,2,1,3))
        for bad in (good[:-1],[1,*good[1:]],[2,2,*good[2:]],[2,1,2,*good[3:]],[True,*good[1:]],good[:-1]+[-1]):
            with self.assertRaises(ValueError):validate_header(bad,2,1,3)

    def test_both_commits_and_phase_order(self):
        states=[[1,2,2,1,4,0,2,1,2,1,262,1],[1,2,2,1,4,0,1,2,2,1,131,1]]
        events=[[9,1,2,3,4,5,6,7,8,9],[8,1,2,3,0,4,5,6,7,8]]
        self.assertTrue(validate_state(states,events,1,2,2,1))
        states[1][2]=1
        self.assertFalse(validate_state(states,events,1,2,2,1))
        states[1][2]=2;events[0][4],events[0][5]=events[0][5],events[0][4]
        self.assertFalse(validate_state(states,events,1,2,2,1))
        validate_ledger()

    def test_state_persistence_reset_and_update_order(self):
        tokens=raw_tokens()
        for t in tokens:t['q']=[f32(x/math.sqrt(128)) for x in t.pop('raw_q')];t['decay']=t['requested_decay']
        zero=[0.0]*16384;start=initial_state()
        one=step(start,zero,tokens[0]);two=step(one['state'],one['state_bounds'],tokens[1])
        self.assertEqual(two['state'],[0.5*x for x in one['state']])
        self.assertEqual(two['delta'],[0.0]*128)
        forgotten=step(start,zero,tokens[1])
        self.assertFalse(check(forgotten['state'],two['state'],two['state_bounds'])['passed'])
        reset=step(zero,zero,tokens[2]);stale=step(two['state'],two['state_bounds'],tokens[2])
        self.assertFalse(check(stale['state'],reset['state'],reset['state_bounds'])['passed'])
        # Output from the decayed old state must fail the updated-state oracle.
        old=step(start,zero,{**tokens[0],'beta':0.0})
        self.assertFalse(check(old['output'],one['output'],one['output_bounds'])['passed'])
        exact=step(zero,zero,tokens[3])
        for name in ('prediction','delta','state','output'):
            self.assertTrue(check(exact[name],exact[name],exact[name+'_bounds'],True)['passed'])


if __name__=='__main__':unittest.main()
