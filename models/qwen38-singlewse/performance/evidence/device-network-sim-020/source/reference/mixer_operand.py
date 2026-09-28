"""Independent packed BF16/E4M3 operand oracle, not device execution evidence.

Quantization uses the independent finite E4M3 nearest-level reference. The wire
format has five headers and128 data words; every body word retains all original
BF16 bits plus the separately rounded E4M3 code and a high-byte route selector.
"""
import numpy as np
from reference.mlp_oracle import bf16_float,quantize


def prepare(raw,invocation,group,parts):
    raw=np.asarray(raw,dtype=np.uint16)
    if raw.shape!=(128,) or not np.isfinite(bf16_float(raw)).all():raise ValueError('One finite original BF16 group128')
    if parts not in (40,48) or not 0<=group<parts or not 0<invocation<2**32:raise ValueError('Mixer operand identity')
    codes,scales=quantize(raw);scale_bits=int(scales.view(np.uint32)[0]);tag=group<<24
    packet=np.empty(133,np.uint32)
    packet[:5]=[tag|(invocation&65535),tag|(invocation>>16),tag|parts,tag|(scale_bits&65535),tag|(scale_bits>>16)]
    packet[5:]=np.uint32(tag)|(codes.astype(np.uint32)<<16)|raw.astype(np.uint32)
    return packet


def decode(packet,invocation,group,parts):
    packet=np.asarray(packet,dtype=np.uint32)
    if packet.shape!=(133,) or parts not in (40,48) or not 0<=group<parts:raise ValueError('Mixer packet extent or selector')
    if np.any(packet[:5]>>16!=group<<8) or np.any(packet[5:]>>24!=group):raise ValueError('Mixer header/body route tags')
    if (int(packet[0])&65535)|((int(packet[1])&65535)<<16)!=invocation or int(packet[2]&65535)!=parts:
        raise ValueError('Mixer invocation or contraction family')
    bits=(int(packet[3])&65535)|((int(packet[4])&65535)<<16)
    scale=np.array([bits],np.uint32).view(np.float32)[0]
    if not np.isfinite(scale) or scale<=0:raise ValueError('Mixer activation scale')
    raw=(packet[5:]&65535).astype(np.uint16);codes=((packet[5:]>>16)&255).astype(np.uint16)
    if np.any((codes&127)==127):raise ValueError('Nonfinite E4M3 operand')
    native=((codes&127)<<7)|((codes&128)<<8)
    return raw,native,scale
