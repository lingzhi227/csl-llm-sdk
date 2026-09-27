"""Exhaustive integer equivalence plus independent finite IEEE half-value proof."""
import argparse,hashlib,json,math,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
digest=hashlib.sha256()
for word in range(65536):
 low=(word<<8)&65535;low=((low-65536 if low&32768 else low)>>1)&0xbfff
 high=((word-65536 if word&32768 else word)>>1)&0xbf80
 expected=[((word&127)<<7)|((word&128)<<8),((word&32512)>>1)|(word&32768)]
 assert [low,high]==expected;digest.update(struct.pack('<HH',low,high))
finite=0
for code in range(256):
 mag=code&127
 if mag==127:continue
 exponent,mantissa=divmod(mag,8)
 value=(mantissa*2.**-9 if exponent==0 else (1+mantissa/8)*2.**(exponent-7))
 value=math.copysign(value,-1 if code&128 else 1)
 expected=struct.unpack('<H',struct.pack('<e',value/256))[0]
 assert expected==((code&127)<<7)|((code&128)<<8);finite+=1
record=dict(schema='fp8-shift-exhaustive-proof-v1',passed=True,physical=False,packed_patterns=65536,embedded_half_values=131072,
 finite_codes_independently_checked=finite,half_bitstream_sha256=digest.hexdigest(),temporary_bytes_removed_at128pairs=256,
 old_vector_operations=9,new_vector_operations=5,native_instruction='@sar16',source='https://cerebras-sdk-docs-130.netlify.app/csl/language/builtins#sar16',
 scope='Exhaustive host integer identity and independent finite IEEE half encoding. Device instruction, native-dot and compiled-SRAM checks remain separately required; no speed claim.')
with a.output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
