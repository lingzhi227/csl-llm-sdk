import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference.mixer_oracle import project_rows
from reference.mlp_oracle import bf16_bits


class MixerOracleTests(unittest.TestCase):
 def test_complete_5120_fp8_contraction_and_independent_bound(self):
  weights=np.full((2,5120),0x38,np.uint8);inputs=np.full(5120,0x3f80,np.uint16)
  scales=np.full((2,40),0x3f80,np.uint16);scales[1,:]=0x3f00
  r=project_rows(weights,inputs,scales)
  np.testing.assert_array_equal(r['ordered_fp32'],[5120.,2560.])
  np.testing.assert_array_equal(r['bf16'],bf16_bits([5120.,2560.]))
  self.assertTrue(np.all(np.abs(r['ordered_fp32']-r['fp64'])<=r['bound']))

 def test_reduction_order_changes_cancellation_result(self):
  weights=np.zeros((1,384),np.uint16);weights[0,::128]=bf16_bits([2.**25,-2.**25,1.])
  inputs=np.zeros(384,np.uint16);inputs[::128]=0x3f80
  r=project_rows(weights,inputs)
  self.assertEqual(r['ordered_fp32'][0],0.);self.assertEqual(r['fp64'][0],1.)
  self.assertGreaterEqual(r['bound'][0],1.)


if __name__=='__main__':unittest.main()
