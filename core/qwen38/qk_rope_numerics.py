"""Original-input Q/K RMS256 and partial RoPE64 source intervals."""
import math
from .rms_numerics import U,TINY,EPSILON,f32,bits,bf16_rne,bf16_of_real,gamma,close
from .gated_numerics import expanded
from .interval_numerics import quantized_vector,outward_radius,check_interval,output_bf16_spans
from .attention_numerics import exact_sum_products
from .rotary_trig import ABS_GATE
D=256
POLICY={'name':'qk-rms256-rope64-position0to7-v1','head_dim':256,'rotary_dim':64,'theta':10000000,'epsilon_f32':EPSILON,
 'domain':'original finite BF16 abs(Q,K)<=1; offset weights in[-2,0.5]; integer text positions0..7, equal three axes',
 'rms':'FP32 squared mean256+epsilon, inverse sqrt, normalize, multiply1+w.float(), BF16 once after gain',
 'rms_approximation_relative':2**-20,'inverse_relative':2**-20,
 'trig':'static official FP32 inv_freq32; device position product, split-pi/2 reduction, sin13/cos12 Horner; BF16 sin/cos',
 'trig_absolute':ABS_GATE,'angle_error':'one FP32 multiplication U*abs(position*inv_freq), plus explicit reference allowance',
 'rotary':'first64 split-half: independent BF16 products, BF16 sum; tail192 exact BF16 pass-through',
 'source_bound':'original-input RMS perturbation through BF16 interval then trig/product/sum intervals; exact finite BF16 endpoint selection',
 'zero':'exact structural zero has zero source radius; signed zeros follow actual separate eager products/additions',
 'limits':'not arbitrary positions, multimodal axes, projections, GQA, cache composition or full layer'}


def fixtures():
    qw=[(0.,0.125,-0.25,-1.,-2.,0.5)[c%6] for c in range(D)]
    kw=[(0.125,-0.25,-1.,-2.,0.5,0.)[(c*5+2)%6] for c in range(D)]
    rows=[]
    for i in range(10):
        gen,pos=(1,i) if i<8 else (2,i-8)
        q=[((c*(i+3)+i*7)%33-16)/16 for c in range(D)]
        k=[((c*(i+5)+i*11)%31-15)/16 for c in range(D)]
        if i==0:
            q[:64]=[0.]*64;k[:64]=[0.]*64;q[0]=1.;q[32]=0.5;k[1]=0.5;k[33]=-1.
        if i==8:q=[0.]*D;k=[0.]*D
        rows.append(dict(generation=gen,position=pos,q=q,k=k))
    return dict(q_weight=qw,k_weight=kw,tokens=rows)


def normalized(x,w):
    if len(x)!=D or len(w)!=D:raise ValueError('Full head dimension required')
    if any(not math.isfinite(v) or abs(v)>1 or expanded(bf16_of_real(v))!=v for v in x):raise ValueError('Input domain')
    if any(not math.isfinite(v) or not -2<=v<=0.5 or expanded(bf16_of_real(v))!=v for v in w):raise ValueError('Weight domain')
    if not exact_sum_products(x,x):raise ValueError('Fixture needs frozen nonexact-reduction policy')
    square=math.fsum(v*v for v in x);mean=square/D
    if f32(mean)!=mean:raise ValueError('Expected exact power-of-two mean')
    denominator=f32(mean+EPSILON);root=math.sqrt(denominator);inverse=1/root
    inverse_error=((1+2**-20)/(1-2**-20)-1)/root
    pre=[];bounds=[];norm=[]
    for a,wv in zip(x,w):
        gain=1+wv
        if f32(gain)!=gain:raise ValueError('Expected exact offset gain')
        norm.append(a*inverse);center=a*inverse*gain
        error=abs(a)*inverse_error*abs(gain)+gamma(2)*abs(a)*(inverse+inverse_error)*abs(gain)
        pre.append(center);bounds.append(outward_radius(center,error))
    rounded,errors=quantized_vector(pre,bounds)
    return dict(square=square,mean=mean,denominator=denominator,root=root,inverse=inverse,normalized=norm,gained=pre,
                output=rounded,bounds=errors,exact_square=True)


def reference(token,qw,kw,frequencies):
    pos=token['position']
    if not isinstance(pos,int) or not 0<=pos<=7 or len(frequencies)!=32:raise ValueError('Position/frequency profile')
    angle=[f32(pos*f) for f in frequencies]
    ae=[U*abs(pos*f) for f in frequencies]
    sine=[math.sin(x) for x in angle];cosine=[math.cos(x) for x in angle]
    sb=[0. if x==0 else ABS_GATE+e+2**-44 for x,e in zip(angle,ae)]
    cb=[0. if x==0 else ABS_GATE+e+2**-44 for x,e in zip(angle,ae)]
    sq,se=quantized_vector(sine,sb);cq,ce=quantized_vector(cosine,cb)
    heads=[]
    for x,w in ((token['q'],qw),(token['k'],kw)):
        nr=normalized(x,w);y=nr['output'];ye=nr['bounds'];products=[[],[]];pe=[[],[]]
        for c in range(64):
            j=c%32;other=c+32 if c<32 else c-32;rot=-y[other] if c<32 else y[other]
            for index,a,err,b,berr in ((0,y[c],ye[c],cq[j],ce[j]),(1,rot,ye[other],sq[j],se[j])):
                center=a*b;error=abs(a)*berr+err*(abs(b)+berr)
                products[index].append(center);pe[index].append(outward_radius(center,error))
        pq0,qe0=quantized_vector(products[0],pe[0]);pq1,qe1=quantized_vector(products[1],pe[1])
        combined=[a+b for a,b in zip(pq0,pq1)];combined_error=[]
        for a,b,e0,e1 in zip(pq0,pq1,qe0,qe1):
            error=e0+e1
            if f32(a+b)!=a+b:error+=U*(abs(a)+abs(b)+error)
            combined_error.append(outward_radius(a+b,error))
        out,oe=quantized_vector(combined,combined_error);out+=y[64:];oe+=ye[64:]
        heads.append(dict(rms=nr,products=products,product_bf16=[pq0,pq1],output=out,bounds=oe,interval=output_bf16_spans(out,oe)))
    return dict(angle=angle,angle_bounds=ae,sine=sine,cosine=cosine,sine_bf16=sq,cosine_bf16=cq,sine_bounds=se,cosine_bounds=ce,heads=heads)


def validate(token,qw,kw,frequencies,actual,probe_inputs):
    from .rotary_trig import evaluate,analytic_budget
    stats=actual['stats'];stages=actual['rms_stages'];normalized_bits=actual['normalized']
    trig=actual['trig'];tc=actual['trig_casts'];rot=actual['rotary_stages'];pc=actual['product_casts'];output=actual['output']
    checks={};casts={}
    def rel(a,b,r):return close(a,b,[r*abs(v)+TINY for v in b])
    for h,(x,w) in enumerate(((token['q'],qw),(token['k'],kw))):
        s=stats[h*5:(h+1)*5];pre=stages[h*512:(h+1)*512];square=math.fsum(v*v for v in x)
        checks[f'{h}_square']=close([s[0]],[square],[gamma(512)*square+512*TINY])
        checks[f'{h}_mean']=rel([s[1]],[s[0]/256],U)
        checks[f'{h}_denominator']=rel([s[2]],[s[1]+EPSILON],U)
        checks[f'{h}_root']=rel([s[3]],[math.sqrt(s[2])],2**-20)
        checks[f'{h}_inverse']=rel([s[4]],[1/s[3]],2**-20)
        checks[f'{h}_normalized']=rel(pre[:256],[v*s[4] for v in x],U)
        checks[f'{h}_gained']=rel(pre[256:],[v*(1+g) for v,g in zip(pre[:256],w)],gamma(2))
        casts[f'{h}_rms']=normalized_bits[h*256:(h+1)*256]==[bf16_rne(bits(v)) for v in pre[256:]]
        nb=[expanded(v) for v in normalized_bits[h*256:(h+1)*256]]
        rb=rot[h*192:(h+1)*192];pbits=pc[h*128:(h+1)*128]
        expected0=[nb[c]*expanded(tc[32+c%32]) for c in range(64)]
        expected1=[(-nb[c+32] if c<32 else nb[c-32])*expanded(tc[c%32]) for c in range(64)]
        checks[f'{h}_product0']=rel(rb[:64],expected0,U)
        checks[f'{h}_product1']=rel(rb[64:128],expected1,U)
        casts[f'{h}_product_zero_signs']=all(bits(a)==bits(b) for a,b in zip(rb[:128],expected0+expected1) if b==0.0)
        casts[f'{h}_product0']=pbits[:64]==[bf16_rne(bits(v)) for v in rb[:64]]
        casts[f'{h}_product1']=pbits[64:]==[bf16_rne(bits(v)) for v in rb[64:128]]
        checks[f'{h}_sum']=rel(rb[128:],[expanded(a)+expanded(b) for a,b in zip(pbits[:64],pbits[64:])],U)
        sum_expected=[expanded(a)+expanded(b) for a,b in zip(pbits[:64],pbits[64:])]
        casts[f'{h}_sum_zero_signs']=all(bits(a)==bits(b) for a,b in zip(rb[128:],sum_expected) if b==0.0)
        casts[f'{h}_output']=output[h*256:h*256+64]==[bf16_rne(bits(v)) for v in rb[128:]]
        casts[f'{h}_tail']=output[h*256+64:(h+1)*256]==normalized_bits[h*256+64:(h+1)*256]
        casts[f'{h}_position_zero']=token['position']!=0 or [expanded(v) for v in output[h*256:(h+1)*256]]==nb
    angles=trig[0::4];sines=trig[2::4];cosines=trig[3::4]
    checks['angles']=rel(angles,[token['position']*v for v in frequencies],U)
    checks['sine']=close(sines,[math.sin(x) for x in angles],[ABS_GATE+TINY]*32)
    checks['cosine']=close(cosines,[math.cos(x) for x in angles],[ABS_GATE+TINY]*32)
    casts['angle_zero_exact']=all(bits(s)==0 and c==1.0 for x,s,c in zip(angles,sines,cosines) if x==0.0)
    casts['trig_domain']=all(0<=x<=7 for x in angles) and all(abs(x)<0.786 for x in trig[1::4])
    casts['sin_bf16']=tc[:32]==[bf16_rne(bits(x)) for x in sines]
    casts['cos_bf16']=tc[32:]==[bf16_rne(bits(x)) for x in cosines]
    probes=actual['probes'];casts['probe_angles']=probes[0::4]==probe_inputs
    checks['probe_sine']=close(probes[2::4],[math.sin(x) for x in probe_inputs],[ABS_GATE+TINY]*len(probe_inputs))
    checks['probe_cosine']=close(probes[3::4],[math.cos(x) for x in probe_inputs],[ABS_GATE+TINY]*len(probe_inputs))
    casts['probe_zero_exact']=all(bits(s)==0 and c==1.0 for x,s,c in zip(probes[0::4],probes[2::4],probes[3::4]) if x==0.0)
    casts['probe_reduction_domain']=all(abs(x)<0.786 for x in probes[1::4])
    return dict(passed=all(x['passed'] for x in checks.values()) and all(casts.values()),stages=checks,conversions=casts)
