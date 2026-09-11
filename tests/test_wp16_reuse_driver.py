"""Synthetic connected host trace/fault tests; no SDK or original model data."""
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
HERE=ROOT/'examples/wp16/connected'
if not HERE.exists():HERE=ROOT
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
checks=load('connected_checks_under_test',HERE/'checks.py')
with patch.dict(sys.modules,{'checks':checks}):driver=load('connected_driver_under_test',HERE/'driver.py')
from qwen38.mlp_reference_checks import decode,encode,rounded
from qwen38.wp16_reuse_source import build_intervals
from qwen38.wp16_reuse_layout import sequence
from qwen38.wp16_reuse_protocol import (identity,numeric_state,final_numeric_state,mlp_state,
    initial_handoff_state,lifecycle_state,weight_snapshot,tile_snapshot,packet,acknowledgement,
    descriptor,completed_handoff_state)

def guarded(values,*,fp32=False,input_buffer=False):
    out=np.zeros(values.size+2,dtype=np.float32 if fp32 else np.uint32)
    out[0],out[-1]=((77.5,-88.25) if input_buffer else (1234.5,-4321.25)) if fp32 else (0xa55a,0x5aa5)
    out[1:-1]=values;return out

class ConnectedFixture:
    def __init__(self):
        self.weights={role:encode(np.full((128,128 if role=='down_proj' else 112),value,dtype='<f8'))
            for role,value in [('gate_proj',1/128),('up_proj',1/64),('down_proj',1/256)]}
        self.hidden=encode(np.array([[1/4096]*112,[2/4096]*112,[0.]*112],dtype='<f8'))
        self.oracle,_=build_intervals({k.removesuffix('_proj'):v for k,v in self.weights.items()},self.hidden)
        self.numeric={}
        for g in (1,2,3):
            n={};hidden=decode(self.hidden[g-1])
            for role in ('gate','up'):
                v=np.sum(decode(self.weights[role+'_proj'])*hidden,axis=1,dtype=np.float64).astype(np.float32)
                n[role+'_fp32']=v;n[role]=encode(rounded(checks.f64_of_f32(v)))
            gate=decode(n['gate']);upper=decode(n['up'])
            exponential=np.exp(-np.abs(gate)).astype(np.float32)
            sigmoid=np.where(gate<0,exponential.astype(np.float64)/(1+exponential.astype(np.float64)),1/(1+exponential.astype(np.float64))).astype(np.float32)
            n['mlp_exp']=exponential;n['mlp_sigmoid']=sigmoid
            n['mlp_silu_fp32']=(gate*sigmoid.astype(np.float64)).astype(np.float32)
            n['silu']=encode(rounded(checks.f64_of_f32(n['mlp_silu_fp32'])))
            n['mlp_product_fp32']=(decode(n['silu'])*upper).astype(np.float32)
            n['product']=encode(rounded(checks.f64_of_f32(n['mlp_product_fp32'])))
            for count in (64,128):
                n['down'+str(count)]=np.sum(decode(self.weights['down_proj'][:,:count])*decode(n['product'][:count]),axis=1,dtype=np.float64).astype(np.float32)
            n['down']=encode(rounded(checks.f64_of_f32(n['down128'])))
            self.numeric[g]=n
        self.trace=[];c=checks.Checks(self.oracle,self.weights)
        for item in sequence():
            c.before_operation(item)
            if item['kind']=='launch':values=None
            elif item['direction']=='h2d':
                values=driver.upload_values(item,self.weights,self.hidden);c.upload(item,values)
            else:values=self.read(item,c);c.receive(item,values)
            c.after_operation(item);self.trace.append((item,values.copy() if values is not None else None))
        self.report=c.finish()

    def read(self,item,c):
        g,pe,name,phase,tile=(item[k] for k in ('generation','pe','name','phase','tile'))
        if phase=='initialize_epoch0':return np.array({'state':numeric_state(0,pe),'handoff_state':initial_handoff_state(0),'mlp_state':mlp_state(0,pe),'lifecycle_state':lifecycle_state(0,pe,stage='initialize')}[name],dtype=np.uint32)
        if phase=='reset_observation':return c.reset_expected(name,pe)
        if phase=='retention_after_down':
            if name=='lifecycle_state':return np.array(lifecycle_state(g,pe,stage='retention'),dtype=np.uint32)
            if name=='handoff_state':return c.get('handoff_'+str(pe if pe<2 else 2),name,pe).copy()
            if pe==2:return c.get('whole_silu_and_product',name,pe).copy()
            if name in ('weights_storage','input_storage'):return c.get('projection_tile_readback' if pe<2 else 'down_tile_readback',name,pe,0 if pe<2 else 1).copy()
            if name=='mlp_product':return c.get('handoff_2',name,3).copy()
            return c.get('finalize_projections' if pe<2 else 'finalize_down',name,pe).copy()
        if phase=='released_join':
            if name=='state':value=final_numeric_state(g,pe,released=True)
            elif name=='mlp_state':value=mlp_state(g,pe,incoming=3 if pe==2 else 4 if pe==3 else 0,computed=pe>=2,tiles=2 if pe==3 else 0,released=True)
            elif name=='lifecycle_state':value=lifecycle_state(g,pe,stage='released')
            else:
                value=c.get('retention_after_down',name,pe).copy();value[0]=5;value[29]+=1
            return np.array(value,dtype=np.uint32)
        if name=='weights_storage':return c.weight_values[pe].copy()
        n=self.numeric[g]
        if name=='input_storage':
            if pe<2:return guarded(decode(self.hidden[g-1]).astype(np.float32),fp32=True,input_buffer=True)
            offset=64*(tile if tile is not None else 1)
            return guarded(decode(n['product'][offset:offset+64]).astype(np.float32),fp32=True,input_buffer=True)
        if name=='output_storage':return guarded(n[('gate_fp32' if pe==0 else 'up_fp32') if pe<2 else 'down'+str(64*(tile+1) if tile is not None else 128)],fp32=True)
        if name=='bf16_storage':return guarded(n[('gate' if pe==0 else 'up') if pe<2 else 'down'])
        if name=='state':
            value=numeric_state(g,pe,phase=1,tiles=tile+1,consumed=112 if pe<2 else (tile+1)*64) if 'tile_readback' in phase else final_numeric_state(g,pe)
            return np.array(value,dtype=np.uint32)
        if name=='tile_guard_snapshot':return np.array(tile_snapshot(g,pe,tile),dtype=np.uint32)
        if name=='weight_guard_snapshot':return np.array(weight_snapshot(g,pe),dtype=np.uint32)
        if name=='timing':
            result=np.zeros(item['count'],dtype=np.uint32)
            for i in range(2 if pe==3 else 1):result[i*6],result[i*6+3]=1,2
            return result
        if phase.startswith('handoff_'):
            edge=int(phase[-1]);body=n[('gate','up','product')[edge]].astype(np.uint32).tolist()
            if name=='wire_data':value=packet(g,edge,body)
            elif name=='wire_ack':value=acknowledgement(g,edge)
            elif name=='handoff_latched':value=descriptor(g,edge)
            elif name=='handoff_state':value=completed_handoff_state(g,edge,pe,receipt_first=g>1 and pe!=edge,command_before_ack=g==2)
            elif name=='mlp_state':value=mlp_state(g,pe,incoming=(1,3,4)[edge])
            else:return guarded(np.array(body,dtype=np.uint32))
            return np.array(value,dtype=np.uint32)
        if phase=='whole_silu_and_product':
            if name=='mlp_state':return np.array(mlp_state(g,2,incoming=3,computed=True),dtype=np.uint32)
            return guarded(n[name] if name.endswith('_fp32') or name in ('mlp_exp','mlp_sigmoid') else n[name.removeprefix('mlp_')],fp32=name.endswith('_fp32') or name in ('mlp_exp','mlp_sigmoid'))
        raise AssertionError(item)

    def replay(self,mutator=None):
        c=checks.Checks(self.oracle,self.weights)
        for item,original in self.trace:
            c.before_operation(item)
            if original is not None:
                values=original.copy()
                if mutator is not None:mutator(item,values)
                if item['direction']=='h2d':c.upload(item,values)
                else:c.receive(item,values)
            c.after_operation(item)
        c.finish();return c

class ConnectedDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=ConnectedFixture()

    def test_complete_synthetic_three_epoch_trace_and_all_groups(self):
        c=self.fixture.replay()
        self.assertEqual(len(c.observed),548);self.assertEqual(len(c.reports['casts']),15)
        self.assertEqual([len(c.reset_seen[g]) for g in (1,2,3)],[56]*3)
        self.assertEqual([len(c.retention_seen[g]) for g in (1,2,3)],[31]*3)
        self.assertEqual([len(c.release_seen[g]) for g in (1,2,3)],[16]*3)

    def test_each_reset_state_word_and_stale_generation_are_rejected(self):
        for name,positions in [('state',range(20)),('mlp_state',range(20)),('handoff_state',range(34)),('lifecycle_state',range(32)),('weight_guard_snapshot',range(16))]:
            for position in positions:
                def mutate(item,values):
                    if item.get('direction')=='d2h' and item['generation']==2 and item['phase']=='reset_observation' and item['pe']==0 and item['name']==name:values[position]^=1
                with self.assertRaisesRegex(ValueError,'reset'):self.fixture.replay(mutate)

    def test_stale_payload_cast_guard_and_retention_failures(self):
        faults=[('projection_tile_readback','input_storage',0),('finalize_projections','bf16_storage',0),
            ('whole_silu_and_product','mlp_silu',2),('down_tile_readback','weights_storage',3),
            ('down_tile_readback','tile_guard_snapshot',3),('retention_after_down','mlp_product',3),
            ('released_join','lifecycle_state',3)]
        for phase,name,pe in faults:
            def mutate(item,values):
                if item.get('direction')=='d2h' and item['generation']==2 and item['phase']==phase and item['name']==name and item['pe']==pe:values.view(np.uint32)[1]^=1
            with self.assertRaises(ValueError):self.fixture.replay(mutate)

    def test_early_begin_release_missing_reset_and_down_overwrite(self):
        complete=self.fixture.replay()
        for phase,missing in [('begin_generation','release_seen'),('begin_projections','reset_seen'),('release_after_retention','retention_seen')]:
            c=self.fixture.replay();g=2
            item=next(o for o in sequence() if o['generation']==g and o['phase']==phase)
            c.next_operation=item['sequence'];getattr(c,missing)[1 if phase=='begin_generation' else g].pop()
            with self.assertRaises(ValueError):c.before_operation(item)
        c=checks.Checks(self.fixture.oracle,self.fixture.weights)
        item=next(o for o in sequence() if o['generation']==1 and o['phase']=='down_tile' and o['name']=='weights_storage')
        c.awaiting=item;c.pending_weights[3]=(1,0)
        with self.assertRaisesRegex(ValueError,'overwrite'):c.upload(item,driver.upload_values(item,self.fixture.weights,self.fixture.hidden))

    def test_duplicate_operation_and_incomplete_event_join(self):
        c=checks.Checks(self.fixture.oracle,self.fixture.weights);item=sequence()[0];c.before_operation(item)
        with self.assertRaises(ValueError):c.before_operation(item)
        for name,field in [('handoff_state',9),('handoff_latched',0),('wire_ack',15)]:
            def mutate(item,values):
                if item.get('direction')=='d2h' and item['generation']==2 and item['phase']=='handoff_2' and item['pe']==2 and item['name']==name:values[field]^=1
            with self.assertRaises(ValueError):self.fixture.replay(mutate)

    def test_raw_native16_evidence_keeps_upper_slots_and_generation_names(self):
        with tempfile.TemporaryDirectory() as folder:
            evidence=driver.Evidence(Path(folder)/'observations')
            for g in (1,2,3):
                item=next(o for o in sequence() if o['generation']==g and o['phase']=='whole_silu_and_product' and o.get('name')=='mlp_product')
                raw=np.full(130,0xbeef3f80,dtype=np.uint32);evidence.observation(item,raw);evidence.flush()
            self.assertEqual(len(evidence.files),3)
            for name in evidence.files:
                with np.load(evidence.root/name,allow_pickle=False) as archive:
                    self.assertEqual(len(archive.files),1);self.assertIn('_i',archive.files[0]);self.assertIn('_c',archive.files[0])
                    self.assertTrue(np.all(archive[archive.files[0]]==0xbeef3f80))

    def test_timestamp_wrong_width_and_live_delta_are_rejected(self):
        for pe in range(4):
            item=next(o for o in sequence() if o.get('direction')=='d2h' and o['generation']==2 and o['name']=='timing' and o['pe']==pe and o['phase']!='reset_observation')
            for count in (0,3,5,7,276):
                if count==item['count']:continue
                c=checks.Checks(self.fixture.oracle,self.fixture.weights);c.awaiting=item;c.g=2
                with self.assertRaisesRegex(ValueError,'dtype/shape'):c.receive(item,np.zeros(count,dtype=np.uint32))
            for values in (np.zeros(item['count'],dtype=np.uint32),np.full(item['count'],0x10000,dtype=np.uint32)):
                c=checks.Checks(self.fixture.oracle,self.fixture.weights);c.g=2
                with self.assertRaises(ValueError):c.check_timing(values,pe,item['phase'])
        c=checks.Checks(self.fixture.oracle,self.fixture.weights);item=next(o for o in sequence() if o.get('name')=='timing' and o['generation']==2 and o['pe']==0)
        stale=dict(item,generation=1,input_index=0);c.awaiting=item
        with self.assertRaisesRegex(ValueError,'current entered'):c.receive(stale,np.zeros(6,dtype=np.uint32))

    def test_every_live_reset_timestamp_rejects_stale_words(self):
        for pe,count in enumerate((6,6,6,12)):
            item=next(o for o in sequence() if o.get('name')=='timing' and o['generation']==2 and o['pe']==pe and o['phase']=='reset_observation')
            for index in range(count):
                c=checks.Checks(self.fixture.oracle,self.fixture.weights);c.awaiting=item;c.g=2
                values=np.zeros(count,dtype=np.uint32);values[index]=1
                with self.assertRaisesRegex(ValueError,'reset'):c.receive(item,values)

    def test_persistence_failure_cannot_bypass_owned_stop_or_replace_primary(self):
        class BrokenEvidence:
            files={};records=0
            def flush(self):raise OSError('synthetic archive failure')
            def event(self,*args,**kwargs):raise OSError('synthetic journal failure')
            def write(self,*args,**kwargs):raise OSError('synthetic summary failure')
        class Runtime:
            calls=0
            def stop(self):self.calls+=1
        original=ValueError('original current-generation numerical failure')
        for primary in (None,original):
            runner=Runtime();report={};error,stopped=driver.flush_and_stop(BrokenEvidence(),runner,report,primary)
            self.assertEqual(runner.calls,1);self.assertTrue(stopped)
            self.assertIn('pre_stop_flush_error',report);self.assertIn('pre_stop_summary_error',report)
            if primary is not None:self.assertIs(error,primary)
            else:self.assertIsInstance(error,OSError)

    def test_stop_failure_and_unconstructed_runtime_remain_failed(self):
        class Evidence:
            files={};records=0
            def flush(self):pass
            def event(self,*args,**kwargs):pass
            def write(self,*args,**kwargs):pass
        class BrokenRuntime:
            calls=0
            def stop(self):self.calls+=1;raise RuntimeError('synthetic owned stop failure')
        runner=BrokenRuntime();report={};error,stopped=driver.flush_and_stop(Evidence(),runner,report,None)
        self.assertFalse(stopped);self.assertIsInstance(error,RuntimeError);self.assertEqual(runner.calls,1)
        original=ValueError('constructor failed');error,stopped=driver.flush_and_stop(Evidence(),None,{},original)
        self.assertFalse(stopped);self.assertIs(error,original)

if __name__=='__main__':unittest.main()
