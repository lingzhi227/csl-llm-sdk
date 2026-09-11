"""Real driver/check/evidence loop with a synthetic transport, never CSL proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import json,math,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from test_resident_spatial import observed_projection,rounded
from qwen38.resident_spatial import TILES
from qwen38.resident_source import build_intervals
from qwen38.mlp_reference_checks import decode,encode
from qwen38.resident_sdk_driver import execute,Evidence,key
from qwen38.resident_sdk_sequence import expected_state,sequence,accounting

def guarded(values,bf16=False):
    result=np.empty(len(values)+2,np.uint32 if bf16 else np.float32)
    result[0],result[-1]=(0xa55a,0x5aa5) if bf16 else (1234.5,-4321.25)
    result[1:-1]=values;return result

def observations(weights,hidden,g):
    result={};h=decode(hidden[g-1]);projections={}
    for role,root in (('gate',0),('up',2)):
        projections[role]=observed_projection(decode(weights[role]),h,96)
    gate=projections['gate']['bf16'];upper=projections['up']['bf16']
    e=np.array([math.exp(-abs(x)) for x in gate],np.float32)
    sig=np.array([ex/(1+ex) if x<0 else 1/(1+ex) for x,ex in zip(gate,e.astype(np.float64))],np.float32)
    activation=(gate*sig.astype(np.float64)).astype(np.float32)
    silu=rounded(activation.astype(np.float64));prod=(silu*upper).astype(np.float32);product=rounded(prod.astype(np.float64))
    projections['down']=observed_projection(decode(weights['down']),product,64)
    for role,root,n,inp in (('gate',0,96,h),('up',2,96,h),('down',6,64,product)):
        p=projections[role]
        for shard in range(2):
            rank=root+shard
            result[key('input',rank)]=guarded(inp[shard*n:(shard+1)*n])
            result[key('partial',rank)]=guarded(p[f'partial{shard}_fp32'])
            result[key('reduced',rank)]=guarded(p['sum_fp32'])
        result[key('rounded',root)]=guarded(encode(p['bf16']),True)
    result[key('output',4)]=result[key('rounded',6)].copy()
    for name,v in (('gate',gate),('up',upper),('silu',silu),('product',product)):result[key(name,5)]=guarded(encode(v),True)
    for name,v in (('exponential',e),('sigmoid',sig),('activation_fp32',activation),('product_fp32',prod)):result[key(name,5)]=guarded(v)
    result[key('request',4)]=guarded(hidden[g-1],True)
    result[key('broadcast',4)]=h.astype(np.float32);result[key('broadcast',5)]=product.astype(np.float32)
    result[key('state',None)]=np.array([x for rank in range(8) for x in expected_state(rank,g)],np.uint32)
    result[key('timing',None)]=np.array([1,0,0,100+g,0,0]*8,np.uint32)
    return result

class SyntheticTransport:
    def __init__(self,weights,hidden,fault=None):
        self.weights=weights;self.hidden=hidden;self.fault=fault;self.calls=[];self.loaded={};self.g=0;self.stops=0
    def load(self):self.calls.append('load')
    def run(self):self.calls.append('run')
    def stop(self):
        self.stops+=1
        if self.fault=='stop':raise RuntimeError('synthetic stop failure')
    def launch(self,name):
        self.calls.append(name)
        if name=='stage':self.g+=1;self.observed=observations(self.weights,self.hidden,self.g)
    def copy(self,item,value):
        self.calls.append((item['direction'],item['name'],item['generation'],item['rank']))
        if item['direction']=='h2d':
            if item['name']=='weights':self.loaded[item['rank']]=value.copy()
            return
        if item['name']=='weights':value[:]=self.loaded[item['rank']]
        elif item['generation']==0:value[:]=np.array([x for rank in range(8) for x in expected_state(rank,0)],np.uint32)
        else:
            value[:]=self.observed[key(item['name'],item['rank'])]
            if self.g==1:
                if self.fault=='missing_shard' and item['name']=='partial' and item['rank']==1:value[1:-1]=0
                if self.fault=='stale_state' and item['name']=='state':value[1]=0
                if self.fault=='guard' and item['name']=='product':value[0]=0
        if item['bits']==16:value[:]|=np.uint32(0xabcd0000)  # Native high halves must be preserved in raw evidence, ignored numerically.

class LoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(1701)
        cls.weights={r:encode(rng.integers(-16,17,(128,n)).astype(np.float64)/64) for r,n in (('gate',192),('up',192),('down',128))}
        cls.hidden=encode(np.vstack([rng.integers(-32,33,192)/32,rng.integers(-32,33,192)/32,np.zeros(192)]))
        cls.source=build_intervals(cls.weights,cls.hidden)
    def test_actual_full_loop_retains_weights_and_persists_all_current_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence=Evidence(Path(tmp)/'evidence');backend=SyntheticTransport(self.weights,self.hidden)
            result=execute(backend,self.weights,self.hidden,self.source,evidence)
            self.assertEqual(result['status'],'passed');self.assertEqual(backend.stops,1)
            self.assertEqual(result['physical_counts']['copies'],121);self.assertEqual(len(evidence.files),5)
            self.assertEqual(sum(len(f['arrays']) for f in evidence.files),112)
            self.assertEqual(sum(a['bytes'] for f in evidence.files for a in f['arrays'].values()),316760)
            self.assertLess(evidence.journal_bytes+evidence.file_bytes,2097152)
            self.assertEqual([x for x in backend.calls if x=='stage'],['stage']*3)
            self.assertTrue(all(c[2]==0 for c in backend.calls if isinstance(c,tuple) and c[:2]==('h2d','weights')))
    def test_real_checks_fail_before_next_input_and_stop_once(self):
        for fault in ('missing_shard','stale_state','guard'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as tmp:
                evidence=Evidence(Path(tmp)/'evidence');backend=SyntheticTransport(self.weights,self.hidden,fault)
                with self.assertRaises(ValueError):execute(backend,self.weights,self.hidden,self.source,evidence)
                self.assertEqual(backend.stops,1);self.assertEqual(backend.g,1)
                self.assertNotIn(('h2d','request',2,4),backend.calls)
                self.assertTrue((Path(tmp)/'evidence/generation1.npz').exists())
    def test_journal_failure_does_not_skip_stop_and_preserves_first_failure(self):
        class BrokenJournal(Evidence):
            def event(self,kind,**fields):
                if kind=='stop_enter':raise OSError('synthetic journal failure')
                super().event(kind,**fields)
        with tempfile.TemporaryDirectory() as tmp:
            backend=SyntheticTransport(self.weights,self.hidden,'guard');evidence=BrokenJournal(Path(tmp)/'evidence')
            with self.assertRaisesRegex(ValueError,'guards'):execute(backend,self.weights,self.hidden,self.source,evidence)
            self.assertEqual(backend.stops,1)
            result=json.loads((Path(tmp)/'result.json').read_text());self.assertEqual(result['error_type'],'ValueError')
    def test_persistence_failure_blocks_reuse_without_overwriting_archive(self):
        class FailPersist(Evidence):
            def event(self,kind,**fields):
                if kind=='group_persisted' and fields['group']=='generation1':raise OSError('synthetic fsync receipt failure')
                super().event(kind,**fields)
        with tempfile.TemporaryDirectory() as tmp:
            backend=SyntheticTransport(self.weights,self.hidden);evidence=FailPersist(Path(tmp)/'evidence')
            with self.assertRaisesRegex(OSError,'receipt'):execute(backend,self.weights,self.hidden,self.source,evidence)
            self.assertEqual(backend.stops,1);self.assertEqual(backend.g,1)
            self.assertEqual(evidence.flush_started,{'initial','generation1'})

if __name__=='__main__':unittest.main()
