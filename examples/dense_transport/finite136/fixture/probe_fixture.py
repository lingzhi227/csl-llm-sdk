"""Exact-dyadic synthetic operands for a future dense transport runtime.

Pure source until separately admitted on the workstation. No SDK or model
weights are imported here. Integer witnesses deliberately make every local FMA
and collective addition exact in FP32; this qualifies transport/layout of the
accepted arithmetic source, not a new floating-point domain.
"""
import struct


ROWS = 128
COLUMNS = 96
LANES = 4
DENOMINATOR = 4096


def f32_bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def bf16_rne_bits(value):
    raw = f32_bits(value)
    assert raw & 0x7f800000 != 0x7f800000
    result = (raw + 0x7fff + ((raw >> 16) & 1)) >> 16
    assert result & 0x7f80 != 0x7f80
    return result


def weight_numerator(phase, y, ordinal, row, column):
    assert 0 <= phase < 2 and 0 <= y < 4 and 0 <= ordinal < LANES
    assert 0 <= row < ROWS and 0 <= column < COLUMNS
    return (17*row + 13*column + 29*ordinal + 7*y + 11*phase) % 255 - 127


def input_numerator(position, phase, ordinal, column):
    assert position in (0, 1) and phase in (0, 1)
    assert 0 <= ordinal < LANES and 0 <= column < COLUMNS
    return (19*column + 7*ordinal + 11*phase + 23*position) % 63 - 31


def weight_words(phase, y, ordinal):
    """6144 u32, column-major128 BF16 rows, low halfword before high."""
    result = []
    for column in range(COLUMNS):
        for row in range(0, ROWS, 2):
            low = bf16_rne_bits(weight_numerator(phase,y,ordinal,row,column)/128)
            high = bf16_rne_bits(weight_numerator(phase,y,ordinal,row+1,column)/128)
            result.append(low | (high << 16))
    assert len(result) == 6144
    return result


def frame_words(position):
    """Two384-element periods; bits retain exact source FP32 representation."""
    return [f32_bits(input_numerator(position,phase,ordinal,column)/32)
        for phase in range(2) for ordinal in range(LANES) for column in range(COLUMNS)]


def descriptor(phase, y):
    return [phase, y, 10, (y+phase+1)%4, ROWS, COLUMNS]


def expected_row(position, phase, y, row):
    """Independent exact-integer witness for all four partials and final row.

    Every product numerator is at most3937, every local prefix at most377952,
    and every four-lane reduction prefix at most1511808. They are below2**24
    on the shared2**-12 lattice. Thus all operations are exactly representable
    FP32, including the prescribed96-FMA and right-to-left collective order.
    """
    partial = []
    for ordinal in range(LANES):
        total = 0
        for column in range(COLUMNS):
            total += weight_numerator(phase,y,ordinal,row,column)*input_numerator(position,phase,ordinal,column)
            assert abs(total) <= 127*31*COLUMNS < (1 << 24)
        partial.append(total)
    reduction = partial[-1]
    for value in reversed(partial[:-1]):
        reduction += value
        assert abs(reduction) <= 127*31*COLUMNS*LANES < (1 << 24)
    return dict(partial_bits=[f32_bits(x/DENOMINATOR) for x in partial],
        result_bits=f32_bits(reduction/DENOMINATOR),
        rounded_bf16=bf16_rne_bits(reduction/DENOMINATOR))


def metadata():
    return dict(scope='Source-only finite synthetic exact-dyadic fixture; no device execution',
        parameters_bytes=786432, frame_bytes_per_serial=3072,
        weight_numerator_bounds=[-127,127], weight_denominator=128,
        input_numerator_bounds=[-31,31], input_denominator=32,
        product_lattice_denominator=DENOMINATOR,
        maximum_local_absolute_prefix_numerator=127*31*COLUMNS,
        maximum_collective_absolute_prefix_numerator=127*31*COLUMNS*LANES,
        exact_FP32_bound=1 << 24, runtime_admitted=False,
        serials=[dict(serial=1,generation=1,position=0),
                 dict(serial=2,generation=1,position=1),
                 dict(serial=3,generation=2,position=0)],
        parameter_retention='Upload once; unchanged across continuation and reset',
        original_model_weights=False, expanded_arithmetic_qualification=False)
