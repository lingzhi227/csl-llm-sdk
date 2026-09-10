"""Selected128Q/128K/128V preprocessing, with device-stage conditional oracles."""
import math
from .rms_numerics import f32,bits,from_bits,bf16_rne,bf16_of_real,gamma,U,TINY,EPSILON,close,PROBE_BITS
from .gated_numerics import expanded,silu
CHANNELS=384
POLICY={'name':'selected-head-preprocess384-bf16-v2','channels':384,'history_width':4,'epsilon':EPSILON,
 'scope':'one matching128Q/128K/128V head, synthetic projected inputs; not10240 channels, projections or layer',
 'convolution':'oldest-to-newest four BF16 history values and asymmetric BF16 weights; four FP32 accumulation terms, BF16 output RNE; history stores pre-convolution inputs',
 'activation':'stable FP32 SiLU of BF16 convolution output then BF16 RNE',
 'qk':'promote BF16 activation to FP32; L2 uses sum(x*x)+eps, no mean; q then divided by sqrt128',
 'gates':'beta=BF16 RNE(sigmoid(BF16 b)), then exact FP32 expansion; t=FP32(a.float()+BF16dtbias); g=-exp(Alog.float())*softplus(t), decay=exp(g)',
 'softplus':'max(t,0)+log1p(exp(-abs(t))); log1p(e) approximated by2*(r+r^3/3+...+r^13/13), r=e/(2+e), r<=1/3; seven terms, no cancellation at negative t',
 'domain':'BF16 input abs<=1, convweight abs<=1, a/b abs<=8, Alog abs<=1, dtbias abs<=0.5; finite normal results or zero; no subnormal/overflow correctness claim',
 'exponential':'custom domain[-24,1] range reduction by splitln2, degree8 FP32 Horner polynomial, exact normal power-of-two scale; gate unchanged',
 'gate_materialization':'separate finalize command reads fixed global BF16 offsets into observed FP32slots before scalar gate function',
 'exp_relative':2**-20,'activation_relative':2**-18,'sqrt_relative':2**-20,'inverse_relative':2**-20,
 'log1p_relative':2**-18,'softplus_relative':2**-18,
 'conversion':'exact RNE of actual preceding FP32 bits; upstream mathematical BF16 parity is separate non-gating evidence',
 'stage_propagation':'conv gamma8 absolute products; nonlinear actual-argument gates; L2 propagates gamma256 sum and sqrt/inverse errors from actual BF16 activation inputs; exact stage casts bridge discontinuities',
 'history':'four slots per channel, left-pad zeros; valid length min(generation token count,4), reset zeroes all1536 values'}


def fixtures():
    weights=[f32(a*((c%7)+1)/8) for c in range(CHANNELS) for a in (1/16,-3/32,5/32,7/16)]
    tokens=[]
    for i in range(8):
        gen,token=(1,i+1) if i<6 else (2,i-5)
        x=[((c*(i+3)+i*5)%31-15)/16 for c in range(CHANNELS)]
        if i==0:
            x=[0.0]*CHANNELS;x[256]=1.0
        if i==1:
            x=[0.0]*CHANNELS;x[0]=1.0;x[128]=-0.5
        if i==6:x=[0.0]*CHANNELS
        a=[-8.,-1.,0.,1.,8.,-0.5,0.,0.5][i];b=[-8.,-0.5,0.,0.5,8.,-1.,0.,1.][i]
        tokens.append({'generation':gen,'token':token,'x':x,'a':a,'b':b,'A_log':[-1.,-0.5,0.,0.5,1.,0.,0.,-0.5][i],
                       'dt_bias':[-0.5,0.,0.5,0.,-0.5,0.5,0.,-0.5][i],'zero_qkv':i==6})
    return {'weights':weights,'tokens':tokens}


def history_step(previous,x):
    if len(previous)!=1536 or len(x)!=CHANNELS:raise ValueError('Invalid history dimensions')
    return [v for c in range(CHANNELS) for v in (*previous[c*4+1:c*4+4],x[c])]


def convolution(history,weights):
    expected=[];bounds=[]
    for c in range(CHANNELS):
        terms=[history[c*4+j]*weights[c*4+j] for j in range(4)]
        expected.append(math.fsum(terms));bounds.append(gamma(8)*math.fsum(abs(x) for x in terms)+8*TINY)
    return expected,bounds


def l2_reference(x):
    square=math.fsum(a*a for a in x);sb=gamma(256)*square+256*TINY
    q=square+EPSILON;qb=sb+gamma(2)*(square+EPSILON+sb)+2*TINY
    low=max(EPSILON*(1-U),q-qb);root=math.sqrt(q);lo=math.sqrt(low)
    approx=(1+POLICY['inverse_relative'])/(1-POLICY['sqrt_relative'])-1
    ib=qb/(lo*root*(lo+root))+approx/lo
    norm=[a/root for a in x];bounds=[abs(a)*ib+U*(abs(a)/root+abs(a)*ib)+TINY for a in x]
    return {'square':square,'square_bound':sb,'q':q,'q_bound':qb,'norm':norm,'norm_bounds':bounds}


def reference(history,weights,token):
    conv,_=convolution(history,weights);convbits=[bf16_of_real(x) for x in conv]
    act=[silu(expanded(x)) for x in convbits];actbits=[bf16_of_real(x) for x in act]
    x=[expanded(v) for v in actbits];q=l2_reference(x[:128])['norm'];k=l2_reference(x[128:256])['norm']
    beta=1/(1+math.exp(-token['b']));t=token['a']+token['dt_bias'];soft=max(t,0)+math.log1p(math.exp(-abs(t)))
    g=-math.exp(token['A_log'])*soft
    return {'conv':conv,'conv_bits':convbits,'activation':act,'activation_bits':actbits,'q':[a/math.sqrt(128) for a in q],
            'k':k,'v':x[256:],'beta':expanded(bf16_of_real(beta)),'g':g,'decay':math.exp(g)}


def validate(token,weights,history,observed):
    def relative(a,b,budget):return close(a,b,[budget*abs(x)+TINY for x in b])
    conv,cb=convolution(history,weights);pre=observed['conv'];convbits=observed['conv_bits']
    act=observed['activation'];actbits=observed['activation_bits'];x=[expanded(a) for a in actbits]
    stages={'convolution':close(pre,conv,cb),
      'conv_exp':relative(observed['activation_exp'],[math.exp(-abs(expanded(a))) for a in convbits],POLICY['exp_relative']),
      'silu':relative(act,[silu(expanded(a)) for a in convbits],POLICY['activation_relative'])}
    conversions={'convolution':convbits==[bf16_rne(bits(a)) for a in pre],
                 'activation':actbits==[bf16_rne(bits(a)) for a in act]}
    for name,vector,normalized,stats in (('q',x[:128],observed['q_normalized'],observed['q_stats']),('k',x[128:256],observed['k'],observed['k_stats'])):
        ref=l2_reference(vector);square,denom,root,inverse=stats
        stages[name+'_sum']=close([square],[ref['square']],[ref['square_bound']])
        stages[name+'_denom']=close([denom],[ref['q']],[ref['q_bound']])
        stages[name+'_sqrt']=relative([root],[math.sqrt(denom)],POLICY['sqrt_relative'])
        stages[name+'_inverse']=relative([inverse],[1/root],POLICY['inverse_relative'])
        stages[name+'_normalized']=close(normalized,ref['norm'],ref['norm_bounds'])
        stages[name+'_actual_inverse']=relative(normalized,[a*inverse for a in vector],U)
    stages['q_scale']=relative(observed['q'],[a/math.sqrt(128) for a in observed['q_normalized']],gamma(2))
    t,eb,beta,ea,et,ratio,logp,soft,g,decay,promoted=observed['gates'][:11]
    stages['gate_add']=relative([t],[token['a']+token['dt_bias']],U)
    stages['beta_exp']=relative([eb],[math.exp(-abs(token['b']))],POLICY['exp_relative'])
    stages['beta_sigmoid']=relative([beta],[1/(1+math.exp(-token['b']))],POLICY['activation_relative'])
    stages['A_exp']=relative([ea],[math.exp(token['A_log'])],POLICY['exp_relative'])
    stages['softplus_exp']=relative([et],[math.exp(-abs(t))],POLICY['exp_relative'])
    stages['log1p_ratio']=relative([ratio],[et/(2+et)],gamma(2))
    stages['log1p']=relative([logp],[math.log1p(et)],POLICY['log1p_relative'])
    stages['softplus']=relative([soft],[max(t,0)+math.log1p(math.exp(-abs(t)))],POLICY['softplus_relative'])
    stages['g_actual_operands']=relative([g],[-ea*soft],U)
    stages['decay_actual_g']=relative([decay],[math.exp(g)],POLICY['exp_relative'])
    exp_arguments=[-abs(expanded(x)) for x in convbits]+[-abs(observed['gates'][12]),observed['gates'][13],-abs(t),g]
    conversions['actual_exp_arguments_in_domain']=all(-24<=x<=1 for x in exp_arguments)
    conversions['gate_input_identity']=observed['gates'][11:]==[token[n] for n in ('a','b','A_log','dt_bias')]
    conversions['beta']=observed['beta_bits']==bf16_rne(bits(beta)) and promoted==expanded(observed['beta_bits'])
    v_exact=observed['v']==x[256:]
    oracle=reference(history,weights,token)
    zero_ok=not token['zero_qkv'] or all(a==0 for n in ('q','k','v') for a in observed[n])
    return {'passed':all(v['passed'] for v in stages.values()) and all(conversions.values()) and v_exact and zero_ok,
      'stages':stages,'conversions':conversions,'v_exact':v_exact,'reset_zero_qkv':zero_ok,
      'conv_oracle_bit_mismatches_non_gating':sum(a!=b for a,b in zip(convbits,oracle['conv_bits'])),
      'activation_oracle_bit_mismatches_non_gating':sum(a!=b for a,b in zip(actbits,oracle['activation_bits']))}
