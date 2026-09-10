import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.rotary_trig import evaluate,analytic_budget,ABS_GATE
from qwen38.qk_rope_numerics import fixtures,reference

class RotaryTrigTests(unittest.TestCase):
    def test_quadrants_and_declared_domain(self):
        for x in (0.,math.pi/4,math.pi/2,math.pi,3*math.pi/2,2*math.pi,7.):
            from qwen38.rms_numerics import f32
            (s,c),detail=evaluate(x)
            self.assertLess(abs(s-math.sin(f32(x))),ABS_GATE)
            self.assertLess(abs(c-math.cos(f32(x))),ABS_GATE)
            self.assertLess(abs(detail['reduced']),0.786)
        for invalid in (-0.01,7.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):evaluate(invalid)
        self.assertTrue(analytic_budget()['passed'])

    def test_identity_tail_and_zero_with_negative_gain(self):
        data=fixtures();frequencies=[1.0/(2**i) for i in range(32)]
        for t in (data['tokens'][0],data['tokens'][8]):
            out=reference(t,data['q_weight'],data['k_weight'],frequencies)
            for h in out['heads']:
                self.assertEqual(h['output'],h['rms']['output'])
                self.assertEqual(h['bounds'][64:],h['rms']['bounds'][64:])
                if t['generation']==2:self.assertEqual(h['output'],[0.]*256)

if __name__=='__main__':unittest.main()
