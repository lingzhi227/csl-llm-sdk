"""Host-only independent RNE check for a direct IEEE-bit E4M3FN encoder candidate.

This does not qualify CSL instruction behavior or dynamic quantization's scale.
"""
import argparse,hashlib,json,math,random,struct
from pathlib import Path

def f32(bits):return struct.unpack('<f',struct.pack('<I',bits))[0]
def bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def encode_bits(word):
 magnitude=word&0x7fffffff;sign=(word>>24)&128
 if magnitude>=0x7f800000:raise ValueError('Finite input required')
 if magnitude>=bits(448.0):return sign|126
 if magnitude<bits(2**-6):return sign|(bits(f32(magnitude)+16384.0)-bits(16384.0))
 return sign|(((magnitude+0x7ffff+((magnitude>>20)&1))>>20)-960)
def value(code):
 e,m=divmod(code,8)
 return m*2**-9 if e==0 else (1+m/8)*2**(e-7)
LEVELS=[value(c) for c in range(127)]
def oracle(word):
 v=f32(word);a=abs(v);sign=128 if word>>31 else 0
 if not math.isfinite(v):raise ValueError('Finite input required')
 if a>=448:return sign|126
 return sign|min(range(127),key=lambda c:(abs(a-LEVELS[c]),c&1))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 words={0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff}
 for level in LEVELS:
  b=bits(level)
  for delta in (-1,0,1):
   if b+delta>=0:words.update([b+delta,(b+delta)|0x80000000])
 for left,right in zip(LEVELS,LEVELS[1:]):
  b=bits((left+right)/2)
  for delta in (-1,0,1):words.update([b+delta,(b+delta)|0x80000000])
 for exp in range(255):
  for mantissa in (0,1,0x3fffff,0x7ffffe,0x7fffff):
   b=(exp<<23)|mantissa;words.update([b,b|0x80000000])
 rng=random.Random(271828)
 for _ in range(50000):
  b=rng.getrandbits(32)
  if (b&0x7f800000)!=0x7f800000:words.add(b)
 for word in words:
  actual=encode_bits(word);expected=oracle(word)
  if actual!=expected:raise AssertionError((hex(word),actual,expected))
 for word in [0x7f800000,0xff800000,0x7fc00000,0xffc00000]:
  try:encode_bits(word)
  except ValueError:continue
  raise AssertionError('Nonfinite accepted')
 source=Path(__file__).resolve().parents[1]/'csl/fp8_encode.csl'
 r=dict(passed=True,physical=False,simulator=False,full_model=False,words_tested=len(words),all126_finite_midpoints_and_neighbors=True,all254_finite_codes_and_neighbors=True,random_seed=271828,
  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Host mathematical direct-bit encoder candidate compared with independent nearest finite E4M3 level and ties-to-even oracle',
  unqualified=['CSL compilation and WSE arithmetic behavior','Instruction cycles','Dynamic scale computation and division convention','Complete model output'])
 with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(r))
if __name__=='__main__':main()
