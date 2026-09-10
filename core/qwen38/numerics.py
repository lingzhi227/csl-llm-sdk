"""Independent small CPU oracle and predeclared FP32 dot-product bounds."""
import math
import struct

ROWS, COLUMNS = 128, 112
POLICY = {'name':'fp32-fma-dot-gamma112-v1','unit_roundoff':2**-24,
          'terms':112,'absolute_floor':112*2**-126,
          'bound':'gamma112 * sum(abs(w_j*x_j)) + absolute_floor',
          'zero_case':'exact numerical zero; signed zero permitted',
          'one_hot_case':'exact FP32 bits of the selected BF16 column',
          'weight_domain':'finite BF16; subnormal BF16 forbidden',
          'input_domain':'finite normal FP32 or zero, absolute value <= 1'}


def weights_from_bits(raw):
    if len(raw) != ROWS*COLUMNS*2:
        raise ValueError('Expected exactly 128x112 BF16 words')
    weights=[]
    for (bits,) in struct.iter_unpack('<H',raw):
        if (bits & 0x7f80)==0x7f80 or ((bits & 0x7f80)==0 and (bits & 0x7f)!=0):
            raise ValueError('Nonfinite or subnormal BF16 outside numerical policy')
        weights.append(struct.unpack('<f',struct.pack('<I',bits<<16))[0])
    return weights


def inputs():
    return [('bounded',[(j%17-8)/16 for j in range(COLUMNS)]),
            ('changed',[((j*7)%23-11)/16 for j in range(COLUMNS)]),
            ('last_column',[0.0]*(COLUMNS-1)+[1.0]),
            ('zero_after_nonzero',[0.0]*COLUMNS)]


def reference(weights, vector):
    if len(weights)!=ROWS*COLUMNS or len(vector)!=COLUMNS:
        raise ValueError('Dot-product shape mismatch')
    if any(not math.isfinite(x) or abs(x)>1 or (x!=0 and abs(x)<2**-126) for x in vector):
        raise ValueError('Input outside policy')
    u=POLICY['unit_roundoff'];gamma=COLUMNS*u/(1-COLUMNS*u)
    expected=[];bounds=[]
    for r in range(ROWS):
        terms=[weights[r*COLUMNS+j]*vector[j] for j in range(COLUMNS)]
        expected.append(math.fsum(terms))
        bounds.append(gamma*math.fsum(abs(term) for term in terms)+POLICY['absolute_floor'])
    return expected,bounds


def evaluate(actual, expected, bounds, case):
    if len(actual)!=ROWS or len(expected)!=ROWS or len(bounds)!=ROWS:
        raise ValueError('Output shape mismatch')
    errors=[]
    for a,e,b in zip(actual,expected,bounds):
        if not math.isfinite(a) or not math.isfinite(e) or not math.isfinite(b) or b<=0:
            raise ValueError('Invalid numerical observation')
        errors.append(abs(a-e))
    passed=all(error<=bound for error,bound in zip(errors,bounds))
    exact=None
    if case=='last_column':
        exact=all(struct.pack('<f',a)==struct.pack('<f',e) for a,e in zip(actual,expected))
        passed=passed and exact
    if case=='zero_after_nonzero':
        exact=all(a==0 for a in actual)
        passed=passed and exact
    return {'passed':passed,'max_absolute_error':max(errors),
            'max_error_over_bound':max(e/b for e,b in zip(errors,bounds)),
            'exact_special_case':exact}
