"""Bounded D256 attention: explicit BF16 source boundaries and stage gates."""
import math
from fractions import Fraction
from .rms_numerics import U,TINY,gamma,f32,bits,bf16_rne,bf16_of_real,close
from .gated_numerics import expanded
from .interval_numerics import quantized_vector,quantized_interval,check_interval,outward_radius,output_bf16_spans
D=256;CAPACITY=8
POLICY={'name':'attention256-cache8-bf16-v1','head_dim':D,'capacity':CAPACITY,
 'input_boundary':'direct synthetic BF16 post-QK-normalization/rotation Q/K, original V and raw gate; no candidate host preprocessing',
 'domain':'abs(Q,K)<=0.5, abs(V)<=1, abs(gate)<=8; only finite normal arithmetic or exact zero; observed exp arguments must lie[-24,0]',
 'score':'FP32 dot256 -> BF16 RNE -> multiply1/16 -> BF16 RNE, matching eager BF16 matmul/scaling outputs',
 'softmax':'max over valid BF16 scaled scores (including all-negative); FP32 subtract/exp/sum/inverse/product; BF16 RNE probabilities before V reduction',
 'output':'FP32 BF16prob*BF16V sum -> BF16 RNE; sigmoid(raw BF16 gate) -> BF16 RNE; product -> BF16 RNE; no SiLU factor',
 'exp_relative':2**-20,'inverse_relative':2**-20,'sigmoid_relative':2**-18,
 'dot_bound':'gamma512*sum(abs products), or data-specific rational proof of exact representable products/all partial sums for source intervals',
 'source_bound':'source-only exp/sum/inverse/probability perturbation; exact-rational BF16 intervals; propagate probability intervals through V reduction and two later BF16 boundaries',
 'conditional_bound':'actual score casts/probabilities/gate operands isolate stages; these never replace the original-input oracle',
 'reference_float64':'reuse outward radius2^-40*error +2^-44*absolute expression magnitude; conservative accounting, not whole-program interval proof',
 'reset':'metadata-only invalidation; real stale BF16 cache retained and fully observed; valid prefix only',
 'overflow':'reject expected next token before any cache/stage/commit change; error and rejection count separate from successful commits'}


def fixtures():
    cache=[-((slot%7)+1)/16 for slot in range(CAPACITY) for _ in range(D)]+[0.5+(slot%4)/16 for slot in range(CAPACITY) for _ in range(D)]
    tokens=[]
    for i in range(10):
        gen,index=(1,i+1) if i<8 else (2,i-7)
        q=[((c*(i+3)+i)%17-8)/16 for c in range(D)]
        k=[-((c*(i+1)+i)%8+1)/16 for c in range(D)] if i<4 else [((c*(i+5)+i*3)%17-8)/16 for c in range(D)]
        v=[((c*7+i*11)%31-15)/16 for c in range(D)]
        gate=[(-8.,-2.,-0.5,0.,0.5,2.,8.)[(c+i)%7] for c in range(D)]
        if i==0:q=[0.0]*D;v=[1.0]+[0.0]*(D-1);gate=[0.0]*D
        if i==2:q=[0.0]*D
        if i==3:q=[0.5]*D
        if i==8:q=[0.0]*D;k=[0.0]*D;v=[0.0]*D;gate=[0.0]*D
        tokens.append(dict(generation=gen,token=index,q=q,k=k,v=v,gate=gate))
    return dict(initial_cache=cache,tokens=tokens,overflow_after_call=8)


def append(cache,token):
    slot=token['token']-1
    if not 0<=slot<CAPACITY:raise ValueError('Capacity refusal before mutation')
    out=cache.copy();out[slot*D:(slot+1)*D]=token['k'];out[CAPACITY*D+slot*D:CAPACITY*D+(slot+1)*D]=token['v']
    return out


def exact_sum_products(a,b):
    products=[Fraction.from_float(x)*Fraction.from_float(y) for x,y in zip(a,b)]
    denom=max(x.denominator for x in products)
    total=sum(abs(x.numerator)*(denom//x.denominator) for x in products)
    return denom<=2**126 and denom&(denom-1)==0 and total<=2**24 and all(Fraction.from_float(f32(float(x)))==x for x in products)


def reference(token,cache):
    n=token['token'];q=token['q'];scores=[];score_bounds=[];proof=[]
    if not 1<=n<=CAPACITY or len(cache)!=2*CAPACITY*D or any(len(token[k])!=D for k in ('q','k','v','gate')):raise ValueError('Invalid attention dimensions')
    for key,limit in (('q',0.5),('k',0.5),('v',1),('gate',8)):
        if any(not math.isfinite(x) or abs(x)>limit or expanded(bf16_of_real(x))!=x for x in token[key]):raise ValueError('Input outside declared BF16 domain')
    for slot in range(n):
        k=cache[slot*D:(slot+1)*D];terms=[a*b for a,b in zip(q,k)];value=math.fsum(terms);exact=exact_sum_products(q,k)
        error=0.0 if exact else gamma(512)*math.fsum(abs(x) for x in terms)+512*TINY
        scores.append(value);score_bounds.append(error);proof.append(exact)
    dotq,dqe=quantized_vector(scores,score_bounds)
    scaled,se=quantized_vector([x/16 for x in dotq],[e/16 for e in dqe])
    maximum=max(scaled);max_error=max(se);exps=[];eb=[];shifts=[]
    for x,error in zip(scaled,se):
        shift=x-maximum;de=error+max_error
        if f32(shift)!=shift:de+=U*(abs(x)+abs(maximum))
        shifts.append(shift);v=math.exp(shift);e=v*math.expm1(de)+POLICY['exp_relative']*v*math.exp(de)
        exps.append(v);eb.append(outward_radius(v,e))
    total=math.fsum(exps);total_error=math.fsum(eb)+gamma(n)*math.fsum(abs(x)+e for x,e in zip(exps,eb))+n*TINY
    low=total-total_error
    if low<=0:raise ValueError('Unusable denominator interval')
    inverse=1/total;ie=total_error/(total*low)+POLICY['inverse_relative']/low
    probs=[x*inverse for x in exps]
    pe=[outward_radius(p,abs(x)*ie+e*(inverse+ie)+U*(abs(x)+e)*(inverse+ie)) for x,e,p in zip(exps,eb,probs)]
    pq,pqe=quantized_vector(probs,pe)
    av=[];ave=[]
    for c in range(D):
        v=[cache[CAPACITY*D+slot*D+c] for slot in range(n)];terms=[a*b for a,b in zip(pq,v)];center=math.fsum(terms)
        perturb=math.fsum(abs(b)*e for b,e in zip(v,pqe));magnitude=math.fsum(abs(b)*(abs(a)+e) for a,b,e in zip(pq,v,pqe))
        arithmetic=0.0 if not any(pqe) and exact_sum_products(pq,v) else gamma(2*n)*magnitude+2*n*TINY
        av.append(center);ave.append(outward_radius(center,perturb+arithmetic,math.fsum(abs(x) for x in terms)))
    aq,aqe=quantized_vector(av,ave)
    gate=[1/(1+math.exp(-x)) for x in token['gate']]
    gq,gqe=quantized_vector(gate,[POLICY['sigmoid_relative']*abs(x) for x in gate])
    output=[];oe=[]
    for a,ae,b,be in zip(aq,aqe,gq,gqe):
        val=a*b;err=abs(a)*be+ae*(abs(b)+be)
        if f32(val)!=val:err+=U*(abs(a)+ae)*(abs(b)+be)
        output.append(val);oe.append(outward_radius(val,err))
    final,fe=quantized_vector(output,oe)
    return dict(scores=scores,score_bounds=score_bounds,score_exact=proof,dot_bf16=dotq,scaled_bf16=scaled,scaled_bounds=se,
      shifts=shifts,exp=exps,probability=probs,probability_bf16=pq,probability_bounds=pqe,attention=av,attention_bf16=aq,attention_bounds=aqe,
      gate_bf16=gq,output=final,output_bounds=fe,output_interval=output_bf16_spans(final,fe))


def validate(token,cache,observed):
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
    source=reference(token,cache);final=[expanded(v) for v in fb]
    return dict(passed=all(x['passed'] for x in checks.values()) and all(conversions.values()),stages=checks,conversions=conversions,
      source_output=check_interval(final,source['output'],source['output_bounds']),nominal_mismatches=sum(a!=bf16_of_real(b) for a,b in zip(fb,source['output'])))
