"""Independent temporal/rounding invariants for the native frontend reference."""
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reference.frontend_oracle import bits,expand,bf,frontend_step,gated_interval


class FrontendOracleTests(unittest.TestCase):
    def setUp(self):
        self.weights=bits(np.tile([1.,2.,3.,4.],(640,1)))
        self.params=bits(np.zeros(6));self.gain=bits(np.ones(128))
        self.history=np.zeros((640,3),np.uint16)

    def test_zero_channels_preserve_nonzero_gate_semantics(self):
        packet,bound,_=frontend_step(np.zeros(1030,np.uint16),self.weights,self.params,self.gain,self.history)
        np.testing.assert_array_equal(packet[:,:384],0)
        np.testing.assert_allclose(packet[:,384:],.5,rtol=0,atol=1e-15)
        self.assertTrue(np.isfinite(bound).all());self.assertTrue((bound>=0).all())
        np.testing.assert_array_equal(self.history,0)

    def test_four_original_taps_and_history_retirement(self):
        original=self.weights.copy();observed=[]
        for position,tap in enumerate([4.,3.,2.,1.,0.]):
            raw=np.zeros(1030,np.uint16)
            if position==0:raw[0]=raw[256]=bits(1.)
            packet,bound,history=frontend_step(raw,self.weights,self.params,self.gain,self.history)
            expected=float(bf(tap/(1+np.exp(-tap))))
            self.assertLessEqual(abs(packet[0,256]-expected),bound[0,256])
            np.testing.assert_array_equal(packet[0,:256],packet[1,:256])
            np.testing.assert_array_equal(packet[0,:256],packet[2,:256])
            np.testing.assert_array_equal(packet[1:,256:384],0)
            observed.append(history.copy())
        np.testing.assert_array_equal(observed[0][0],[0,0,bits(1.)])
        np.testing.assert_array_equal(observed[1][0],[0,bits(1.),0])
        np.testing.assert_array_equal(observed[2][0],[bits(1.),0,0])
        np.testing.assert_array_equal(observed[3],0)
        np.testing.assert_array_equal(self.weights,original)

    def test_reset_replay_is_exact_but_continuation_keeps_history(self):
        raw=bits(np.linspace(-.25,.25,1030));first=frontend_step(raw,self.weights,self.params,self.gain,self.history)[0]
        continued=frontend_step(raw,self.weights,self.params,self.gain,self.history)[0]
        self.assertGreater(np.linalg.norm(continued-first),0)
        self.history.fill(0)
        replay=frontend_step(raw,self.weights,self.params,self.gain,self.history)[0]
        np.testing.assert_array_equal(replay,first)

    def test_gated_interval_encloses_independent_constant_case(self):
        core=bits(np.full(128,.5));z=bits(np.ones(128))
        lo,hi=gated_interval(core,z,self.gain)
        normalized=bf(.5/np.sqrt(.25+1e-6))
        expected=bf(bf(normalized)*1/(1+np.exp(-1)))
        self.assertTrue(np.all(lo<=expected) and np.all(expected<=hi))
        self.assertLess(np.linalg.norm(hi-lo),.03*np.sqrt(128)*abs(expected))
        lo,hi=gated_interval(core,bits(np.zeros(128)),self.gain)
        self.assertTrue(np.all(lo<=0) and np.all(hi>=0))


if __name__=='__main__':unittest.main()
