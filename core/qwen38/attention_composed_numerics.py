"""Independent original-input QK-to-attention profile; accepted standalone policies unchanged."""
import math
from .rms_numerics import U,TINY,gamma,f32,bits,bf16_rne,bf16_of_real,close
from .gated_numerics import expanded
from .interval_numerics import quantized_vector,check_interval,outward_radius,output_bf16_spans
from .attention_numerics import POLICY as ATTENTION_POLICY,fixtures as attention_fixtures,append,exact_sum_products
from .qk_rope_numerics import POLICY as QK_POLICY,fixtures as qk_fixtures,reference as qk_reference
D=256;CAPACITY=8
POLICY={'name':'original-qk-to-attention256-cache8-position0to7-v1',
 'source':'original BF16 Q/K/V/rawgate, offset weights, static official FP32 inv_freq and text position; no observed intermediate enters source oracle',
 'qk_policy':QK_POLICY,'exp_relative':ATTENTION_POLICY['exp_relative'],'inverse_relative':ATTENTION_POLICY['inverse_relative'],'sigmoid_relative':ATTENTION_POLICY['sigmoid_relative'],
 'component_profile':'abs post-RMS/rotary source Q/K enclosure<=3 is necessary but not sufficient; coupled valid-score gap<=24 required separately for every prefix',
 'source_uncertainty':'propagate current Q and each persistent cached K radius through dot, two BF16 score casts, max/exp/sum/inverse/probability, BF16 probability, V reduction, sigmoid BF16 and product BF16',
 'correlations':'triangle bounds treat shared-weight/token errors as independent over-approximations; no cancellation of unknown correlated errors',
 'zero':'use exact-rational dot proof when both operand radii vanish; exact structural zero stays zero',
 'conditional':'same accepted arithmetic stage gates on actual operands; no call to old0.5-domain source oracle',
 'limits':'specific synthetic one Q/K pair, fullD256/cache8, positions0..7; not arbitrary vectors with abs<=3, projections, GQA, full layer or long context'}


def fixtures():
    qk=qk_fixtures();att=attention_fixtures();rows=[]
    for a,b in zip(qk['tokens'],att['tokens']):
        if a['generation']!=b['generation'] or a['position']+1!=b['token']:raise ValueError('Original fixture alignment')
        rows.append({**a,'token':b['token'],'v':b['v'],'gate':b['gate']})
    return dict(q_weight=qk['q_weight'],k_weight=qk['k_weight'],initial_cache=att['initial_cache'],tokens=rows,overflow_after_call=8)


def source_step(data,token,cache,cache_bounds,frequencies):
    pre=qk_reference(token,data['q_weight'],data['k_weight'],frequencies)
    q,k=[h['output'] for h in pre['heads']];qe,ke=[h['bounds'] for h in pre['heads']]
    derived={**token,'q':q,'k':k};updated=append(cache,derived);errors=cache_bounds.copy();slot=token['token']-1
    errors[slot*D:(slot+1)*D]=ke
    attention=attention_source(derived,updated,qe,errors)
    return dict(preprocessing=pre,derived=derived,cache=updated,cache_bounds=errors,attention=attention)


def attention_source(token,cache,q_bounds,cache_bounds):
    n=token['token'];q=token['q'];scores=[];score_bounds=[];proof=[]
    if not 1<=n<=CAPACITY or len(cache)!=2*CAPACITY*D or any(len(token[k])!=D for k in ('q','k','v','gate')):raise ValueError('Invalid attention dimensions')
    for key,limit in (('q',3),('k',3),('v',1),('gate',8)):
        if any(not math.isfinite(x) or abs(x)>limit or expanded(bf16_of_real(x))!=x for x in token[key]):raise ValueError('Input outside declared BF16 domain')
    if len(q_bounds)!=D or len(cache_bounds)!=2*CAPACITY*D or any(not math.isfinite(v) or v<0 for v in q_bounds+cache_bounds):raise ValueError('Source uncertainty shape/domain')
    if any(cache_bounds[CAPACITY*D:]):raise ValueError('Original V must be exact BF16')
    if any(abs(v)+e>3 for v,e in zip(q,q_bounds)):raise ValueError('Q source enclosure exceeds component profile')
    if any(abs(v)+e>3 for v,e in zip(cache[:n*D],cache_bounds[:n*D])):raise ValueError('K source enclosure exceeds component profile')
    for slot in range(n):
        k=cache[slot*D:(slot+1)*D];ke=cache_bounds[slot*D:(slot+1)*D]
        terms=[a*b for a,b in zip(q,k)];value=math.fsum(terms)
        exact=not any(q_bounds) and not any(ke) and exact_sum_products(q,k)
        perturb=math.fsum(abs(a)*be+ae*(abs(b)+be) for a,ae,b,be in zip(q,q_bounds,k,ke))
        magnitude=math.fsum((abs(a)+ae)*(abs(b)+be) for a,ae,b,be in zip(q,q_bounds,k,ke))
        error=0.0 if exact else outward_radius(value,perturb+gamma(512)*magnitude+512*TINY)
        scores.append(value);score_bounds.append(error);proof.append(exact)
    dotq,dqe=quantized_vector(scores,score_bounds)
    scaled,se=quantized_vector([x/16 for x in dotq],[e/16 for e in dqe])
    gap_upper=max(v+e for v,e in zip(scaled,se))-min(v-e for v,e in zip(scaled,se))
    if gap_upper>24:raise ValueError('Coupled original-source score-gap exceeds qualified exp domain')
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
    return dict(score_gap_upper=gap_upper,scores=scores,score_bounds=score_bounds,score_exact=proof,dot_bf16=dotq,scaled_bf16=scaled,scaled_bounds=se,
      shifts=shifts,exp=exps,probability=probs,probability_bf16=pq,probability_bounds=pqe,attention=av,attention_bf16=aq,attention_bounds=aqe,
      gate_bf16=gq,output=final,output_bounds=fe,output_interval=output_bf16_spans(final,fe))

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
