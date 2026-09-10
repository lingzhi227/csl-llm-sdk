import math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.preprocess_numerics import fixtures,history_step,convolution,l2_reference
from qwen38.preprocess_protocol import validate_ledger


class PreprocessTests(unittest.TestCase):
    def test_causal_orientation_history_and_reset(self):
        data=fixtures();history=[0.0]*1536
        first=history_step(history,data['tokens'][0]['x']);conv,_=convolution(first,data['weights'])
        self.assertEqual(conv[256],data['weights'][256*4+3])
        self.assertNotEqual(conv[256],data['weights'][256*4])
        for i in range(6):
            history=history_step(history,data['tokens'][i]['x'])
            for c in (0,128,256,383):
                past=[t['x'][c] for t in data['tokens'][max(0,i-3):i+1]]
                self.assertEqual(history[c*4:(c+1)*4],[0.0]*(4-len(past))+past)
        reset=history_step([0.0]*1536,data['tokens'][6]['x'])
        self.assertEqual(reset,[0.0]*1536)
        self.assertNotEqual(history_step(history,data['tokens'][6]['x']),reset)
        validate_ledger()

    def test_l2_sum_epsilon_not_mean(self):
        ref=l2_reference([1.0]*128)
        self.assertLess(abs(ref['norm'][0]-1/math.sqrt(128+1e-6)),1e-10)
        self.assertLess(ref['norm'][0],0.1)
        zero=l2_reference([0.0]*128);self.assertEqual(zero['norm'],[0.0]*128);self.assertGreater(zero['q'],0)

    def test_log1p_truncation_is_only_part_of_budget(self):
        for e in (math.exp(-8.5),0.01,0.1,0.5,1.0):
            r=e/(2+e);partial=2*sum(r**(2*j+1)/(2*j+1) for j in range(7))
            remainder=2*r**15/(15*(1-r*r))
            self.assertLessEqual(abs(math.log1p(e)-partial),remainder+1e-15)


if __name__=='__main__':unittest.main()
