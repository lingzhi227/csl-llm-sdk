"""Exact three-field wire separation and original shared quantization oracle."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.mixer_operand import prepare,decode
from reference.mlp_oracle import bf16_bits,quantize,fp8_float


class MixerOperandTests(unittest.TestCase):
    def test_all_finite_codes_all_selectors_have_exact_native_float_values(self):
        codes=np.array([c for c in range(256) if c&127!=127],np.uint32)
        for selector in range(48):
            # Model only the documented integer vector instructions. Compare
            # their floating value to an independent exponent/mantissa oracle.
            upper=((selector<<8)|codes).astype(np.uint16)
            shifted=(upper<<8).view(np.int16)
            native=((shifted>>1).view(np.uint16)&0xbfff)
            np.testing.assert_array_equal(native,((codes&127)<<7)|((codes&128)<<8))
            np.testing.assert_array_equal(native.view(np.float16).astype(np.float64)*256,fp8_float(codes.astype(np.uint8)))
            self.assertTrue(np.all((selector<<8<=upper)&(upper<=(selector<<8)|255)))
        raw=np.arange(65536,dtype=np.uint32)
        for selector,code in [(0,0),(39,126),(47,254)]:
            word=np.uint32(selector<<24|code<<16)|raw
            np.testing.assert_array_equal(word.astype(np.uint16),raw.astype(np.uint16))
            self.assertTrue(np.all(word>>24==selector));self.assertTrue(np.all(((word>>16)&255)==code))

    def test_original_quantization_and_bf16_survive_zero_extremes_and_changed_input(self):
        cases=[np.zeros(128,np.uint16),bf16_bits(np.linspace(-448,448,128,dtype=np.float32)),
               bf16_bits(np.sin(np.arange(128,dtype=np.float32))*.1),
               np.tile(np.array([0,32768,1,32769,0x7f7f,0xff7f,0x3f80,0xbf80],np.uint16),16)]
        for parts,group in [(40,39),(48,47)]:
            for invocation,raw in enumerate(cases,1):
                packet=prepare(raw,invocation,group,parts);observed,native,scale=decode(packet,invocation,group,parts)
                codes,expected=quantize(raw)
                np.testing.assert_array_equal(observed,raw)
                np.testing.assert_array_equal(native.view(np.float16).astype(np.float64)*256,fp8_float(codes))
                self.assertEqual(scale.view(np.uint32),expected[0].view(np.uint32));self.assertEqual(packet.size,133)

    def test_wrong_epoch_tag_family_scale_and_code_are_rejected(self):
        original=prepare(np.zeros(128,np.uint16),0x12345678,39,40)
        for kind in ('epoch','header_tag','body_tag','family','scale','code','size'):
            packet=original.copy()
            if kind=='epoch':packet[0]^=1
            elif kind=='header_tag':packet[4]^=1<<24
            elif kind=='body_tag':packet[5]^=1<<24
            elif kind=='family':packet[2]^=1
            elif kind=='scale':packet[3]&=np.uint32(0xffff0000);packet[4]&=np.uint32(0xffff0000)
            elif kind=='code':packet[5]|=127<<16
            else:packet=packet[:-1]
            with self.assertRaises(ValueError,msg=kind):decode(packet,0x12345678,39,40)


if __name__=='__main__':unittest.main()
