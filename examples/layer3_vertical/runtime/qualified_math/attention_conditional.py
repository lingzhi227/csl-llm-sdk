"""Accepted attention gates on actual operands, independent of narrow fixture oracle."""
import math
from .rms_numerics import U,TINY,gamma,bits,bf16_rne,close,from_bits
D=256;CAPACITY=8
POLICY=dict(exp_relative=2**-20,inverse_relative=2**-20,sigmoid_relative=2**-18)
def expanded(raw):return from_bits(raw<<16)

def validate_attention(token,cache,observed):
    n=token['token'];score=observed['scores'];casts=observed['score_casts'];stats=observed['stats'];stage=observed['stages'];outcasts=observed['output_casts']
    def rel(a,b,r):return close(a,b,[r*abs(x)+TINY for x in b])
    dot=[];db=[]
    for slot in range(n):
        terms=[a*b for a,b in zip(token['q'],cache[slot*D:(slot+1)*D])];dot.append(math.fsum(terms));db.append(gamma(512)*math.fsum(abs(x) for x in terms)+512*TINY)
    dot_pre=score[:8];scaled_pre=score[8:16];shift=score[16:24];exps=score[24:32];prob=score[32:40]
    dot_bits=casts[:8];scaled_bits=casts[8:16];prob_bits=casts[16:24];scaled=[expanded(v) for v in scaled_bits[:n]]
    maximum,total,inverse=stats
    checks={'dot':close(dot_pre[:n],dot,db),'scale':rel(scaled_pre[:n],[expanded(v)/16 for v in dot_bits[:n]],U),
      'shift':rel(shift[:n],[x-maximum for x in scaled],U),'exponential':rel(exps[:n],[math.exp(x) for x in shift[:n]],POLICY['exp_relative']),
      'sum':close([total],[math.fsum(exps[:n])],[gamma(n)*math.fsum(abs(x) for x in exps[:n])+n*TINY]),
      'inverse':rel([inverse],[1/total],POLICY['inverse_relative']),
      'probability':rel(prob[:n],[x*inverse for x in exps[:n]],U)}
    conversions={'dot':dot_bits[:n]==[bf16_rne(bits(x)) for x in dot_pre[:n]],'scaled':scaled_bits[:n]==[bf16_rne(bits(x)) for x in scaled_pre[:n]],
      'probability':prob_bits[:n]==[bf16_rne(bits(x)) for x in prob[:n]],'maximum_valid':maximum==max(scaled),
      'invalid_slots_zero':all(x==0 for seq in [score[i*8:(i+1)*8] for i in range(5)]+[casts[i*8:(i+1)*8] for i in range(3)] for x in seq[n:]),
      'actual_exp_domain':all(-24<=x<=0 for x in shift[:n])}
    av=stage[:D];gate_input=stage[D:2*D];ge=stage[2*D:3*D];gp=stage[3*D:4*D];fp=stage[4*D:5*D]
    ab=outcasts[:D];gb=outcasts[D:2*D];fb=outcasts[2*D:3*D]
    expected=[];bounds=[];pv=[expanded(v) for v in prob_bits[:n]]
    for col in range(D):
        terms=[pv[t]*cache[CAPACITY*D+t*D+col] for t in range(n)];expected.append(math.fsum(terms));bounds.append(gamma(2*n)*math.fsum(abs(x) for x in terms)+2*n*TINY)
    checks['value_reduction']=close(av,expected,bounds)
    checks['gate_exp']=rel(ge,[math.exp(-abs(x)) for x in token['gate']],POLICY['exp_relative'])
    checks['sigmoid']=rel(gp,[1/(1+math.exp(-x)) for x in token['gate']],POLICY['sigmoid_relative'])
    checks['product']=rel(fp,[expanded(a)*expanded(b) for a,b in zip(ab,gb)],U)
    conversions.update(gate_input=gate_input==token['gate'],gate_exp_domain=all(-24<=-abs(x)<=0 for x in gate_input),
      attention=ab==[bf16_rne(bits(x)) for x in av],gate=gb==[bf16_rne(bits(x)) for x in gp],output=fb==[bf16_rne(bits(x)) for x in fp])
    return dict(passed=all(x['passed'] for x in checks.values()) and all(conversions.values()),stages=checks,conversions=conversions)
