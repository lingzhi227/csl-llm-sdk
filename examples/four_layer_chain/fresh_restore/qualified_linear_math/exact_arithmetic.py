"""Qualified integer binary32 arithmetic; fixed samples check every physical matrix lane."""
def check(value,message):
 if not value:raise ValueError(message)

def rounded_integer(value, power):
    """IEEE binary32 RNE on an exact integer times 2**power."""
    magnitude = abs(value)
    if not magnitude:
        return 0
    drop = max(0, magnitude.bit_length() - 24, -149 - power)
    if drop:
        quotient, remainder = divmod(magnitude, 1 << drop)
        half = 1 << (drop - 1)
        quotient += remainder > half or (remainder == half and quotient & 1)
        magnitude = quotient << drop
    return -magnitude if value < 0 else magnitude

def word32(value, power):
    if not value:
        return 0
    sign = int(value < 0) << 31
    magnitude = abs(value)
    top = magnitude.bit_length() - 1
    exponent = top + power
    check(-126 <= exponent <= 127, 'Fixture normal output domain')
    shift = top - 23
    if shift > 0:
        check(magnitude % (1 << shift) == 0, 'Already rounded FP32 integer')
        mantissa = magnitude >> shift
    else:
        mantissa = magnitude << -shift
    return sign | ((exponent + 127) << 23) | (mantissa & 0x7fffff)

def components32(bits):
    exponent = (bits >> 23) & 255
    check(exponent != 255, 'Finite observation')
    if exponent == 0:
        check(bits & 0x7fffffff == 0, 'No subnormal fixture operand')
        return 0, 0
    mantissa = (bits & 0x7fffff) | (1 << 23)
    return (-mantissa if bits >> 31 else mantissa), exponent - 150

def multiply32(a, b):
    va, pa = components32(int(a)); vb, pb = components32(int(b))
    if va == 0 or vb == 0:
        return (int(a) ^ int(b)) & 0x80000000
    return word32(rounded_integer(va * vb, pa + pb), pa + pb)

def exact_dot(ww, xx):
    terms = []
    for w, x in zip(ww, xx):
        w, x = int(w), int(x)
        if w & 32767 == 0 or x & 32767 == 0:
            continue
        ew, ex = (w >> 7) & 255, (x >> 7) & 255
        assert 0 < ew < 255 and 0 < ex < 255
        value = ((w & 127) + 128) * ((x & 127) + 128)
        if (w ^ x) & 32768: value = -value
        terms.append((value, ew + ex - 268))
    if not terms: return 0
    power = min(p for _, p in terms); acc = 0
    for value, p in terms:
        acc = rounded_integer(acc + (value << (p-power)), power)
        # This independent exact sample does not assume a hardware FTZ choice.
        assert acc == 0 or abs(acc).bit_length()-1+power >= -126
    return word32(acc, power)
