"""Source-only QK retained-statistics/enclosure adaptation; no overwritten RMS observation."""
import math
from .rms_numerics import U,TINY,EPSILON,gamma,bits,bf16_rne,close,from_bits
from .qk_alias_enclosure import check_normalized
from .qk_retained_stats import check_retained_stats
ABS_GATE=2**-20
def expanded(raw):return from_bits(raw<<16)

def validate_qk(token,qw,kw,frequencies,actual,*,input_bits,gain_bits):
    stats=actual['stats'];normalized_bits=actual['normalized']
    if 'rms_stages' in actual:
        raise ValueError('Overwritten RMS products cannot be supplied as observations')
    if len(input_bits)!=512 or len(gain_bits)!=512 or len(normalized_bits)!=512:
        raise ValueError('All original raw512-channel inputs/gains/normalized values required')
    if [bits(v) for v in token['q']+token['k']] != [v<<16 for v in input_bits]:
        raise ValueError('QK actual operand raw identity')
    if [bits(v) for v in qw+kw] != [v<<16 for v in gain_bits]:
        raise ValueError('QK actual gain raw identity')
    stat_bits=[bits(v) for v in stats]
    rms=check_retained_stats(input_bits,stat_bits)
    norm_passed=True;ambiguous=0;zero_sign_unchecked=0;failed_samples=[]
    for index in range(512):
        gate=check_normalized(input_bits[index],gain_bits[index],stat_bits[5*(index//256)+4],normalized_bits[index])
        norm_passed=norm_passed and gate['passed']
        ambiguous+=int(not gate['single_numeric_rounding_bin'])
        zero_sign_unchecked+=int(normalized_bits[index]&0x7fff==0)
        if not gate['passed'] and len(failed_samples)<8:
            failed_samples.append(dict(index=index,**gate))
    trig=actual['trig'];tc=actual['trig_casts'];rot=actual['rotary_stages'];pc=actual['product_casts'];output=actual['output']
    checks={};casts={}
    def rel(a,b,r):return close(a,b,[r*abs(v)+TINY for v in b])
    for h,(x,w) in enumerate(((token['q'],qw),(token['k'],kw))):
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
    # Standalone probes are archived for diagnosis; this gate covers the neural trig path.
    return dict(passed=rms['passed'] and norm_passed and all(x['passed'] for x in checks.values()) and all(casts.values()),
        retained_RMS=rms,normalization_enclosure=dict(passed=norm_passed,values=512,ambiguous_numeric_bins=ambiguous,
            zero_sign_not_bitwise_checked=zero_sign_unchecked,failed_samples=failed_samples,FP32_intermediates_observed=False),
        stages=checks,conversions=casts,rotary_and_trig_gates_unchanged=True)
