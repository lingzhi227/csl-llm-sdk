"""Decode actual committed QK archive bits; this is not numerical acceptance."""
import struct

MAGIC = 0x514B4131
ARRAYS = {
    'trig': (0, 512, 'f32'),
    'rotary_stages': (512, 2048, 'f32'),
    'product_casts': (2048, 2560, 'u16'),
    'trig_casts': (2560, 2688, 'u16'),
    'probes': (2688, 2816, 'f32'),
    'normalized': (2816, 3840, 'u16'),
}


def decode_archive(status, raw_u32, *, head, identity, generation=1):
    if not (0 <= head < 24 and len(identity) == 5 and generation > 0):
        raise ValueError('Exact expected archive owner and generation')
    if not (len(status) == 16 and len(raw_u32) == 960
            and all(type(v) is int and 0 <= v <= 0xffffffff for v in [*status, *raw_u32])):
        raise ValueError('Exact unsigned archive record widths')
    expected = [2 * generation, MAGIC, 1 | (head << 16), generation,
                *identity, 960, 704, 256, 2]
    if not (status[:13] == expected and status[15] == 2 * generation
            and 0 < status[13] <= 65 and status[14] == 0):
        raise ValueError('Actual sink commit, identity, block lengths and zero errors required')
    payload = struct.pack('<960I', *raw_u32)
    arrays = {}
    for name, (start, stop, dtype) in ARRAYS.items():
        width, code = (4, 'I') if dtype == 'f32' else (2, 'H')
        bits = list(struct.unpack('<' + str((stop - start) // width) + code, payload[start:stop]))
        arrays[name] = dict(dtype=dtype, bits=bits)
    return dict(head=head, sink=[748, 24-head], identity=list(identity), generation=generation,
                raw_status_u32=status, archive_marker_sequence=status[13], arrays=arrays,
                actual_sink_committed=True, raw_bytes=3840, numerical_acceptance=False)
