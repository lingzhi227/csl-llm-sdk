from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.numerics import weights_from_bits,inputs,reference,evaluate


class NumericsTests(unittest.TestCase):
    def test_last_column_and_zero_exactness(self):
        raw=b''.join(struct.pack('<H',0x3f80 if c==111 else 0) for r in range(128) for c in range(112))
        weights=weights_from_bits(raw)
        case,x=inputs()[2];expected,bounds=reference(weights,x)
        self.assertEqual(expected,[1.0]*128)
        self.assertTrue(evaluate(expected,expected,bounds,case)['passed'])
        changed=expected.copy();changed[0]=1.0000001192092896
        self.assertFalse(evaluate(changed,expected,bounds,case)['passed'])
        case,x=inputs()[3];expected,bounds=reference(weights,x)
        self.assertTrue(evaluate(expected,expected,bounds,case)['passed'])
        changed=expected.copy();changed[0]=1e-40
        self.assertFalse(evaluate(changed,expected,bounds,case)['passed'])

    def test_nonfinite_weights_and_nan_result_rejected(self):
        with self.assertRaises(ValueError):weights_from_bits(struct.pack('<H',0x7fc1)*(128*112))
        with self.assertRaises(ValueError):evaluate([float('nan')]*128,[0.0]*128,[1.0]*128,'bounded')

    def test_cancellation_uses_absolute_product_scale(self):
        weights=[1.0]*(128*112);x=[0.5,-0.5]*56
        expected,bounds=reference(weights,x)
        self.assertEqual(expected,[0.0]*128)
        self.assertGreater(bounds[0],0)
        self.assertTrue(evaluate([bounds[0]/2]*128,expected,bounds,'bounded')['passed'])
        self.assertFalse(evaluate([bounds[0]*2]*128,expected,bounds,'bounded')['passed'])
