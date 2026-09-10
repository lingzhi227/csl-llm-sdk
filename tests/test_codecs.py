from array import array
from dataclasses import replace
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.codecs import encode, decode


class CodecTests(unittest.TestCase):
    def test_bf16_special_bits_and_odd_offset(self):
        values = array('H', [123, 0, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1])
        view = memoryview(values)[1:]
        for codec in ('memcpy16_containers', 'packed_u32_stream'):
            frame = encode(view, codec)
            self.assertEqual(decode(frame, codec), list(view))
            self.assertEqual(frame.logical_bytes, 14)
            self.assertEqual(frame.sdk_count, 7 if codec == 'memcpy16_containers' else 4)
        self.assertEqual(encode(array('H', [0x1234, 0xabcd]), 'packed_u32_stream').payload,
                         bytes.fromhex('3412cdab'))

    def test_raw_bits(self):
        values = array('I', [0, 0x80000000, 0xffffffff, 0x7fc00001])
        self.assertEqual(decode(encode(values, 'raw_u32'), 'raw_u32'), list(values))

    def test_reject_wrong_dtype_empty_and_stride(self):
        for values in (array('f', [1]), array('i', [1]), array('H'), memoryview(array('H', [1,2,3]))[::2]):
            with self.assertRaises(ValueError):
                encode(values, 'memcpy16_containers')
        with self.assertRaises(OverflowError):
            array('H', [65536])

    def test_unit_size_codec_and_padding_rejection(self):
        frame = encode(array('H', [1]), 'packed_u32_stream')
        for bad in (replace(frame, sdk_count=True), replace(frame, logical_count=0),
                    replace(frame, payload=b''), replace(frame, sdk_unit='bytes'),
                    replace(frame, payload=struct.pack('<I', 65537))):
            with self.assertRaises(ValueError):
                decode(bad, 'packed_u32_stream')
        with self.assertRaises(ValueError):
            decode(frame, 'memcpy16_containers')

    def test_native_d2h_upper_halfword_is_unspecified(self):
        frame = encode(array('H', [0x8000]), 'memcpy16_containers')
        self.assertEqual(decode(replace(frame, payload=struct.pack('<I', 0xabcd8000)), frame.codec), [0x8000])
