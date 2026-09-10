"""Independent original-slab dot oracle for the declared valid prefix."""
import math,struct


def policy(columns):
    if type(columns) is not int or not 1<=columns<=5120:raise ValueError('Invalid column count')
    return {'name':'persistent-direct-fma-v1','columns':columns,'unit_roundoff':2**-24,
            'absolute_floor_per_term':2**-126,
            'order':'FMA each valid column in increasing global index, preserve output across tiles',
            'bound':'gamma(consumed)*sum(abs(w*x)) + consumed*min_normal',
            'padding':'BF16 +1 weights and FP32 +7 inputs; excluded by valid count',
            'zero_case':'exact numerical zero, signed zero allowed',
            'onehot_case':'exact numerical value of selected BF16 column; signed zero allowed',
            'domain':'finite normal BF16 or zero; finite dyadic FP32 inputs within [-1,1] for valid columns'}


def decode(raw):
    if len(raw)!=128*5120*2:raise ValueError('Wrong slab byte count')
    result=[]
    for (bits,) in struct.iter_unpack('<H',raw):
        if (bits&0x7f80)==0x7f80 or ((bits&0x7f80)==0 and (bits&0x7f)!=0):
            raise ValueError('Nonfinite/subnormal BF16 outside policy')
        result.append(struct.unpack('<f',struct.pack('<I',bits<<16))[0])
    return result


def vectors(columns):
    return [('bounded',[(j%17-8)/16 for j in range(columns)]),
            ('changed',[((j*7)%23-11)/16 for j in range(columns)]),
            ('last_valid',[0.0]*(columns-1)+[1.0]),
            ('zero_after_nonzero',[0.0]*columns)]


def reference(weights,vector,consumed):
    if len(weights)!=128*5120 or not 0<consumed<=len(vector)<=5120:raise ValueError('Reference shape/count mismatch')
    u=2**-24;gamma=consumed*u/(1-consumed*u);values=[];bounds=[]
    for row in range(128):
        terms=[weights[row*5120+j]*vector[j] for j in range(consumed)]
        values.append(math.fsum(terms));bounds.append(gamma*math.fsum(abs(x) for x in terms)+consumed*2**-126)
    return values,bounds


def check(actual,expected,bounds,exact=False):
    if len(actual)!=128 or len(expected)!=128 or len(bounds)!=128:raise ValueError('Observation size mismatch')
    if any(not math.isfinite(x) for seq in (actual,expected,bounds) for x in seq):raise ValueError('Nonfinite observation')
    if any(b<=0 for b in bounds):raise ValueError('Invalid bound')
    errors=[abs(a-e) for a,e in zip(actual,expected)]
    passed=all(e<=b for e,b in zip(errors,bounds)) and (not exact or actual==expected)
    return {'passed':passed,'max_absolute_error':max(errors),'max_error_over_bound':max(e/b for e,b in zip(errors,bounds)),
            'exact_case':exact}
