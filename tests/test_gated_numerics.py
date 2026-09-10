import copy,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.gated_numerics import fixtures,reference,validate,f32,bits,bf16_rne,expanded,silu


def rounded_stages(case):
    x,w,z=[case[n] for n in ('x','w','z')]
    square=0.0
    for a in x:square=f32(square+f32(a*a))
    q=f32(f32(square/128)+f32(1e-6));root=f32(math.sqrt(q));inverse=f32(1/root)
    norm=[f32(a*inverse) for a in x];nb=[bf16_rne(bits(a)) for a in norm]
    gp=[f32(expanded(a)*b) for a,b in zip(nb,w)];gb=[bf16_rne(bits(a)) for a in gp]
    gate=[f32(silu(a)) for a in z];pre=[f32(expanded(a)*b) for a,b in zip(gb,gate)]
    return {'stats':[square,q,root,inverse],'norm':norm,'norm_bits':nb,'gain_pre':gp,'gain_bits':gb,
        'exp':[f32(math.exp(-abs(a))) for a in z],'gate':gate,'precast':pre,'output_bits':[bf16_rne(bits(a)) for a in pre]}


class GatedTests(unittest.TestCase):
    def test_rounding_stages_and_boundary_mutation(self):
        for c in fixtures():self.assertTrue(validate(c,rounded_stages(c))['passed'])
        c=fixtures()[-1];good=rounded_stages(c)
        for field in ('norm_bits','gain_bits','output_bits'):
            bad=copy.deepcopy(good);bad[field][7]^=1
            self.assertFalse(validate(c,bad)['passed'],field)

    def test_gate_semantics_and_adversaries(self):
        c=fixtures()[-1];ref=reference(c['x'],c['w'],c['z'])
        self.assertGreater(ref['adversaries']['late_cast'],0)
        bad=rounded_stages(c);bad['gate']=[1/(1+math.exp(-a)) for a in c['z']]
        self.assertFalse(validate(c,bad)['passed'])
        bad=rounded_stages(c);bad['gain_pre']=[f32(expanded(a)*(1+b)) for a,b in zip(bad['norm_bits'],c['w'])]
        self.assertFalse(validate(c,bad)['passed'])


if __name__=='__main__':unittest.main()
