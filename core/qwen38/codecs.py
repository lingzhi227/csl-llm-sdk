"""Explicit little-endian host wire encodings; never numerical float casts."""
from dataclasses import dataclass
import struct

KINDS = {'memcpy16_containers': ('H', 2), 'packed_u32_stream': ('H', 2),
         'raw_u32': ('I', 4)}


@dataclass(frozen=True)
class WireFrame:
    codec: str
    logical_count: int
    logical_bytes: int
    host_bytes: int
    sdk_count: int
    sdk_unit: str
    payload: bytes


def layout(codec, count):
    if codec not in KINDS or type(count) is not int or count <= 0:
        raise ValueError('Unknown codec or non-positive element count')
    logical_bytes = count * KINDS[codec][1]
    words = (count + 1)//2 if codec == 'packed_u32_stream' else count
    unit = '16bit_elements_in_u32_containers' if codec == 'memcpy16_containers' else 'u32_words'
    return logical_bytes, words * 4, words, unit


def encode(values, codec):
    if codec not in KINDS:
        raise ValueError('Unknown codec')
    view = memoryview(values)
    expected, width = KINDS[codec]
    if view.ndim != 1 or not view.c_contiguous or view.format != expected or view.itemsize != width:
        raise ValueError('Expected contiguous native unsigned H/I bit array; floats are forbidden')
    count = len(view)
    logical_bytes, host_bytes, words, unit = layout(codec, count)
    if codec == 'packed_u32_stream':
        payload = b''.join(struct.pack('<I', view[i] | (view[i+1] << 16 if i+1 < count else 0))
                           for i in range(0, count, 2))
    else:
        payload = b''.join(struct.pack('<I', value) for value in view)
    return WireFrame(codec, count, logical_bytes, host_bytes, words, unit, payload)


def decode(frame, expected_codec):
    if frame.codec != expected_codec:
        raise ValueError('Codec mismatch')
    sizes = layout(frame.codec, frame.logical_count)
    if any(type(v) is not int for v in (frame.logical_bytes, frame.host_bytes, frame.sdk_count)):
        raise ValueError('Counts must be integers')
    if (frame.logical_bytes, frame.host_bytes, frame.sdk_count, frame.sdk_unit) != sizes:
        raise ValueError('Length/unit metadata mismatch')
    if type(frame.payload) is not bytes or len(frame.payload) != frame.host_bytes:
        raise ValueError('Payload byte length mismatch')
    words = [x[0] for x in struct.iter_unpack('<I', frame.payload)]
    if frame.codec == 'raw_u32':
        return words
    if frame.codec == 'memcpy16_containers':
        # Upper halfword on D2H is explicitly ignored; only low 16 bits are valid.
        return [word & 65535 for word in words]
    if frame.logical_count % 2 and words[-1] >> 16:
        raise ValueError('Nonzero packed tail padding')
    return [part for word in words for part in (word & 65535, word >> 16)][:frame.logical_count]
