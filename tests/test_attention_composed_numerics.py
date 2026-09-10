import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.attention_composed_numerics import fixtures,source_step,attention_source

class ComposedAttentionNumericsTests(unittest.TestCase):
    def test_original_first_token_uses_rms_rope_source(self):
        data=fixtures();t=data['tokens'][0]
        out=source_step(data,t,data['initial_cache'],[0.]*4096,[1./2**i for i in range(32)])
        self.assertGreater(max(map(abs,out['derived']['q'])),0.5)
        self.assertEqual(out['attention']['output'],[0.5]+[0.]*255)
        self.assertEqual(out['cache'][:256],out['preprocessing']['heads'][1]['output'])

    def test_component_bound_does_not_replace_coupled_gap(self):
        t=dict(token=2,q=[3.]*256,k=[-3.]*256,v=[0.]*256,gate=[0.]*256)
        cache=[3.]*256+[-3.]*256+[0.]*(4096-512)
        with self.assertRaisesRegex(ValueError,'score-gap'):
            attention_source(t,cache,[0.]*256,[0.]*4096)

    def test_current_q_and_persistent_key_uncertainty_reach_scores(self):
        t=dict(token=2,q=[0.5]*256,k=[-0.5]*256,v=[0.]*256,gate=[0.]*256)
        cache=[0.5]*256+[-0.5]*256+[0.]*(4096-512)
        exact=attention_source(t,cache,[0.]*256,[0.]*4096)
        qe=[0.125]+[0.]*255;ke=[0.,0.125]+[0.]*(4096-2)
        uncertain=attention_source(t,cache,qe,ke)
        self.assertEqual(exact['score_bounds'][0],0.)
        self.assertGreater(uncertain['score_bounds'][0],0.125)
        self.assertGreater(uncertain['scaled_bounds'][0],exact['scaled_bounds'][0])

if __name__=='__main__':unittest.main()
