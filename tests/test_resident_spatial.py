"""Independent tiny arithmetic fixtures; not CSL execution or model acceptance."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import math,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from qwen38.resident_spatial import ledger,TILES
from qwen38.resident_spatial_numerics import pair_bounds,projection,fragment_source
from qwen38.mlp_numerics import round_scalar

def rounded(values):return np.array([round_scalar(float(v)) for v in values])

def observed_projection(weights,hidden,n):
    # Synthetic bounded dyadics only: a BF16 product and FP32 accumulator sum
    # fit exactly in FP64 here, so one FP32 rounding models this fixture's FMA.
    parts=[]
    for shard in range(2):
        acc=np.zeros(128,np.float32)
        for c in range(shard*n,(shard+1)*n):acc=(acc.astype(np.float64)+weights[:,c]*hidden[c]).astype(np.float32)
        parts.append(acc.astype(np.float64))
    total=(parts[0]+parts[1]).astype(np.float32).astype(np.float64)
    return dict(partial0_fp32=parts[0],partial1_fp32=parts[1],sum_fp32=total,bf16=rounded(total))

class SpatialTests(unittest.TestCase):
    def assert_inside(self,bounds,values):
        self.assertTrue(np.all((bounds[:,0]<=values)&(values<=bounds[:,1])))
    def test_dense_changed_and_zero_with_reused_weights(self):
        rng=np.random.default_rng(923)
        weights={k:rng.integers(-16,17,(128,n)).astype(np.float64)/64 for k,n in (('gate',192),('up',192),('down',128))}
        pins={k:v.tobytes() for k,v in weights.items()};outputs=[]
        cases=[rng.integers(-32,33,192).astype(np.float64)/32 for _ in range(2)]+[np.zeros(192,np.float64)]
        for hidden in cases:
            source=fragment_source(weights,hidden);observed={}
            for role in ('gate','up'):
                for key,value in observed_projection(weights[role],hidden,96).items():observed[role+'_'+key]=value
            # Independent host libm ideal is one admissible synthetic arithmetic
            # realization; it does not emulate the device exponential polynomial.
            g=observed['gate_bf16'];u=observed['up_bf16']
            observed['silu_fp32']=np.array([float(np.float32(x/(1+math.exp(-x)))) for x in g])
            observed['silu_bf16']=rounded(observed['silu_fp32'])
            observed['product_fp32']=(observed['silu_bf16']*u).astype(np.float32).astype(np.float64)
            observed['product_bf16']=rounded(observed['product_fp32'])
            for key,value in observed_projection(weights['down'],observed['product_bf16'],64).items():observed['down_'+key]=value
            for key,value in observed.items():
                with self.subTest(array=key):self.assert_inside(source[key],value)
            outputs.append(observed['down_bf16'])
            if not hidden.any():
                for value in source.values():self.assertTrue(np.all(value==0))
                self.assertTrue(np.all(outputs[-1]==0))
        self.assertGreater(np.count_nonzero(outputs[0]!=outputs[1]),100)
        self.assertGreater(np.count_nonzero(outputs[1]),100)
        for k in weights:self.assertEqual(weights[k].tobytes(),pins[k])
    def test_missing_shard_and_early_BF16_partial_cast_are_rejected(self):
        weights=np.zeros((128,192),np.float64);weights[:,[0,1,96]]=1
        hidden=np.zeros(192,np.float64);hidden[0]=1;hidden[[1,96]]=1/256
        source=projection(weights,np.column_stack((hidden,hidden)),96)
        observed=observed_projection(weights,hidden,96)
        self.assert_inside(source['bf16'],observed['bf16'])
        early=rounded(rounded(observed['partial0_fp32'])+rounded(observed['partial1_fp32']))
        missing=rounded(observed['partial0_fp32'])
        self.assertTrue(np.all(early<source['bf16'][:,0]));self.assertTrue(np.all(missing<source['bf16'][:,0]))
    def test_pair_cancellation_subnormal_operands_and_failures(self):
        a=np.array([[1.,1.],[2.**-127,2.**-127]],np.float64)
        b=np.array([[-1.,-1.],[2.**-127,2.**-127]],np.float64)
        value=pair_bounds(a,b)
        self.assertTrue(value[0,0]<=0<=value[0,1]);self.assertTrue(value[1,0]<=0<=value[1,1])
        self.assertTrue(value[1,0]<=2.**-126<=value[1,1])
        with self.assertRaises(ValueError):pair_bounds(a,np.zeros((3,2)))
        with self.assertRaises(ValueError):pair_bounds(np.full((1,2),3.4e38),np.full((1,2),3.4e38))
        with self.assertRaises(ValueError):projection(np.zeros((128,192)),np.zeros((192,2)),64)
    def test_tile_capacity_weight_partition_and_traffic(self):
        plan=ledger();self.assertEqual(len(set((t.x,t.y) for t in TILES)),8)
        self.assertEqual(sum(2*128*t.columns for t in TILES),plan['initial_weight_bytes'])
        self.assertTrue(all(t['proposed_upper_bytes']<=48128 for t in plan['tiles']))
        self.assertEqual(plan['device_control_packets'],7*5)
        self.assertEqual(plan['device_application_packet_words'],7*5*4+9*4+3*64)
        self.assertEqual(plan['host_stage_launches_per_input'],1)
        self.assertEqual(plan['steady_token_weight_H2D_bytes'],0)

if __name__=='__main__':unittest.main()
