"""Independent exhaustive encoding identity; does not assert device behavior.

E4M3FN values can be embedded in IEEE binary16 divided by256 using bit insertion,
without a lookup table. Native FP16 input denormal support must still be measured
before this transformation can enter a WSE kernel.
"""
import argparse
import json
import math
from pathlib import Path
import struct


def fp8(code):
    sign=-1.0 if code&128 else 1.0
    exponent=(code>>3)&15;mantissa=code&7
    if exponent==15 and mantissa==7:raise ValueError('NaN encoding excluded')
    return sign*(math.ldexp(mantissa,-9) if exponent==0 else math.ldexp(1+mantissa/8,exponent-7))


def half_embed(code):
    bits=((code&127)<<7)|((code&128)<<8)
    return struct.unpack('<e',struct.pack('<H',bits))[0]


def prove():
    codes=[c for c in range(256) if (c&127)!=127]
    for c in codes:
        original,scaled=fp8(c),half_embed(c)*256
        assert struct.pack('<f',original)==struct.pack('<f',scaled),c
    for a in codes:
        for b in codes:
            assert fp8(a)*fp8(b)==half_embed(a)*half_embed(b)*65536,(a,b)
    return dict(passed=True,scope='IEEE mathematical encoding identity only; no hardware or model performance claim',
        finite_encodings=len(codes),ordered_products=len(codes)**2,
        fp16_bits='((fp8_code & 127) << 7) | ((fp8_code & 128) << 8)',
        fp16_value_scale=256,fp32_dot_scale=65536,
        accumulation_requirement='Native mixed FP16 multiply / FP32 accumulate with original term order; rescale before original activation/weight scales',
        device_denormal_qualification_required=True,device_kernel_qualified=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=prove()
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))
