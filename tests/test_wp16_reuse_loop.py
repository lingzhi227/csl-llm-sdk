"""Real driver/Checks/Evidence loop with synthetic transport; never import a real SDK."""
from contextlib import ExitStack,redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

from test_wp16_reuse_driver import ConnectedFixture,driver,checks,ROOT
from qwen38.wp16_reuse_layout import ARRAYS,sequence,budget
from qwen38.wp16_reuse_protocol import key,identity
from qwen38 import wp16_reuse_sdk_admission as admission,wp16_affinity as affinity


def archive_name(item):
    return ('g'+str(item['generation'])+'_i'+str(item['input_index'])+'_c'+str(item['generation'])+'_'+item['phase']+
        ('_tile'+str(item['tile']) if item['tile'] is not None else '')+'.npz')


def tag(item):
    return key(item['generation'],item['input_index'],item['generation'],item['phase'],item['name'],item['pe'],item['tile'])


class RealLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fixture=ConnectedFixture()

    def exercise(self,module=driver,*,fault=None,stop_failure=False):
        fixture=self.fixture;trace=fixture.trace;test=self
        with tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
            work=Path(folder);root=work/'device-evidence'
            (work/'physical-sequence.json').write_text(json.dumps(sequence()))
            (work/'resource-plan.json').write_text(json.dumps(budget()))
            for name in ('runtime-admission-simulate.json','container-admission-simulate.json'):(work/name).write_text('{}')
            durable=set();creates=[];synced=[];fd_paths={};barriers=[];raw_arrays={}
            failure=OSError('injected archive '+str(fault)+' failure') if fault else None
            stop_error=RuntimeError('injected owned stop failure')
            target='g1_i0_c1_reset_observation.npz'
            original_open=Path.open;original_fsync=os.fsync

            def durable_group(g,phase,count,tile=None):
                i=identity(g)[1];name='g'+str(g)+'_i'+str(i)+'_c'+str(g)+'_'+phase+('_tile'+str(tile) if tile is not None else '')+'.npz'
                test.assertIn(name,durable)
                with zipfile.ZipFile(root/name) as archive:test.assertEqual(len(archive.infolist()),count)
                barriers.append((runtime.position,name))

            class Runtime:
                position=0;stops=0;loads=0;runs=0
                symbols={name:index+1 for index,name in enumerate(ARRAYS)}
                def get_id(self,name):return self.symbols[name]
                def load(self):self.loads+=1
                def run(self):self.runs+=1
                def stop(self):
                    self.stops+=1
                    if stop_failure:raise stop_error
                def next(self,kind):
                    item,values=trace[self.position];test.assertEqual(item['kind'],kind)
                    self.position+=1;return item,values
                def launch(self,name,**params):
                    item,_=self.next('launch');test.assertEqual(name,item['name']);test.assertEqual(params,dict(nonblock=False))
                def copy(self,direction,symbol,values,pe,y,width,height,count,params):
                    item,expected=trace[self.position]
                    test.assertEqual(item['direction'],direction);test.assertEqual((symbol,pe,count),(self.symbols[item['name']],item['pe'],item['count']))
                    test.assertEqual((y,width,height),(0,1,1));test.assertEqual(params,dict(data_type=item['bits'],streaming=False,order='row',nonblock=False))
                    if direction=='h2d':
                        g=item['generation'];name=item['name'];tile=item['tile']
                        if name=='generation_control' and g>1:durable_group(g-1,'released_join',16)
                        if name=='release_control':durable_group(g,'retention_after_down',31)
                        if name=='input_storage':
                            durable_group(g,'reset_observation',56)
                            if g>1:durable_group(g-1,'projection_tile_readback',8,0)
                        if name=='weights_storage' and pe==3:
                            if tile==1:durable_group(g,'down_tile_readback',5,0)
                            elif g>1:durable_group(g-1,'down_tile_readback',5,1)
                        test.assertEqual(values.dtype,expected.dtype);test.assertEqual(values.tobytes(),expected.tobytes())
                    else:
                        raw=expected.copy()
                        if item['bits']==16:raw|=np.uint32(0xbeef0000)
                        values[:]=raw;raw_arrays[tag(item)]=(item,raw)
                    self.next('copy')
                def memcpy_h2d(self,symbol,values,pe,y,width,height,count,**params):self.copy('h2d',symbol,values,pe,y,width,height,count,params)
                def memcpy_d2h(self,values,symbol,pe,y,width,height,count,**params):self.copy('d2h',symbol,values,pe,y,width,height,count,params)
            runtime=Runtime()

            def opened(path,*args,**kwargs):
                mode=args[0] if args else kwargs.get('mode','r')
                if path.parent==root and path.suffix=='.npz' and 'x' in mode:
                    creates.append(path.name)
                    if fault=='write' and path.name==target:raise failure
                stream=original_open(path,*args,**kwargs)
                if path.parent==root:fd_paths[stream.fileno()]=path
                return stream
            def fsync(fd):
                path=fd_paths.get(fd)
                if path is not None and path.parent==root and path.suffix=='.npz':
                    if fault=='fsync' and path.name==target:raise failure
                    original_fsync(fd);durable.add(path.name);synced.append((runtime.position,path.name));return
                original_fsync(fd)

            modules={name:ModuleType(name) for name in ('cerebras','cerebras.sdk','cerebras.sdk.runtime','cerebras.sdk.runtime.sdkruntimepybind')}
            for name,m in modules.items():m.__path__=[]
            sdk=modules['cerebras.sdk.runtime.sdkruntimepybind']
            sdk.SdkRuntime=lambda *args:runtime;sdk.MemcpyDataType=SimpleNamespace(MEMCPY_16BIT=16,MEMCPY_32BIT=32)
            sdk.MemcpyOrder=SimpleNamespace(ROW_MAJOR='row');sdk.SimfabConfig=lambda **kwargs:kwargs
            sdk.SdkTarget=SimpleNamespace(WSE3='fake-WSE3');sdk.get_platform=lambda *args:'fake-platform'
            stack.enter_context(patch.dict(sys.modules,modules));stack.enter_context(patch.object(admission,'verify',return_value='synthetic-loop-only'))
            stack.enter_context(patch.object(affinity,'current',return_value=[0]));stack.enter_context(patch.object(module,'load_prepared',return_value=(fixture.weights,fixture.hidden,fixture.oracle)))
            stack.enter_context(patch.dict(os.environ,{'WP16_SDK_STAGE':'simulate'}));stack.enter_context(patch.object(Path,'cwd',return_value=work))
            stack.enter_context(patch.object(Path,'open',opened));stack.enter_context(patch.object(os,'fsync',fsync))
            error=None
            with redirect_stdout(io.StringIO()):
                try:module.main()
                except BaseException as exc:error=exc
            result=json.loads((root/'result.json').read_bytes())
            journal=[json.loads(line) for line in (root/'journal.jsonl').read_text().splitlines()]
            test.assertEqual(runtime.stops,1);test.assertEqual(result['normal_stop'],not stop_failure)
            test.assertEqual(sum(r['kind']=='stop_enter' for r in journal),1);test.assertEqual(journal[-1]['kind'],'stop_enter' if stop_failure else 'stop_exit')
            archive_members={}
            for path in root.glob('*.npz'):
                with zipfile.ZipFile(path) as archive:archive_members[path.name]=archive.namelist()
            summary=dict(error=error,failure=failure,stop_error=stop_error,result=result,journal=journal,position=runtime.position,
                creates=creates,durable=durable,synced=synced,barriers=barriers,
                archive_members=archive_members)
            if error is None:
                test.assertEqual(runtime.position,696);test.assertEqual(result['status'],'connected_reuse_passed')
                test.assertEqual(result['physical_counts'],dict(copy_calls=647,launch_calls=49,h2d=99,d2h=548,host_bytes=978168,native_bytes=539652))
                test.assertEqual(len(creates),39);test.assertEqual(len(set(creates)),39);test.assertEqual(len(durable),39)
                test.assertEqual(len(raw_arrays),548);test.assertEqual(sum(a.nbytes for _,a in raw_arrays.values()),660392)
                members=0
                for archive in creates:
                    expected={name:pair for name,pair in raw_arrays.items() if archive_name(pair[0])==archive}
                    with np.load(root/archive,allow_pickle=False) as stored:
                        test.assertEqual(set(stored.files),set(expected));members+=len(stored.files)
                        for name,(item,array) in expected.items():
                            test.assertEqual(stored[name].dtype,array.dtype);test.assertEqual(stored[name].shape,array.shape)
                            test.assertEqual(stored[name].tobytes(),array.tobytes())
                test.assertEqual(members,548)
                for g in (1,2,3):
                    i=identity(g)[1];name='g'+str(g)+'_i'+str(i)+'_c'+str(g)+'_reset_observation.npz'
                    test.assertEqual(len(summary['archive_members'][name]),56)
                    last=next(o['sequence'] for o in sequence() if o['generation']==g and o.get('name')=='mlp_product' and o['pe']==3 and o['phase']=='reset_observation')
                    test.assertEqual([position for position,n in synced if n==name],[last+1])
                test.assertEqual([len(result['checks'][name]) for name in ('reset','retention','released')],[168,93,48])
                pairs=[r for r in journal if r['kind'] in ('copy_enter','copy_exit','launch_enter','launch_exit')]
                test.assertEqual(len(pairs),1392)
                for index,item in enumerate(sequence()):
                    for record in pairs[2*index:2*index+2]:test.assertEqual(record['operation'],item)
            return summary

    def test_real_loop_writes_all39_complete_archives_before_dependent_operations(self):
        result=self.exercise();self.assertIsNone(result['error']);self.assertGreater(len(result['barriers']),25)

    def test_old002_boundary_reproduces_exact_reset_collision(self):
        path=ROOT/'local/wp16-connected-sdk-002/driver.py'
        if not path.exists():path=ROOT/'qualification/driver002-reproduction.py'
        raw=path.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),'b4b1a4f5dbe2fef13be3a80c1a2f3a7d90da60f24b6b44ccaa852b50d64e53da')
        spec=importlib.util.spec_from_file_location('frozen002_driver_loop_reproduction',path);old=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'checks':checks}):spec.loader.exec_module(old)
        result=self.exercise(old)
        self.assertIsInstance(result['error'],FileExistsError);self.assertEqual(result['position'],74)
        self.assertEqual(len(result['archive_members']['g1_i0_c1_reset_observation.npz']),38)
        self.assertEqual(result['creates'].count('g1_i0_c1_reset_observation.npz'),3)
        self.assertEqual(result['result']['status'],'failed')

    def test_archive_write_and_fsync_fault_preserve_primary_stop_and_no_advancement(self):
        last=next(o['sequence'] for o in sequence() if o['generation']==1 and o.get('name')=='mlp_product' and o['pe']==3 and o['phase']=='reset_observation')
        for fault in ('write','fsync'):
            with self.subTest(fault=fault):
                result=self.exercise(fault=fault)
                self.assertIs(result['error'],result['failure']);self.assertEqual(result['position'],last+1)
                self.assertEqual(result['result']['status'],'failed')
                self.assertNotIn('g1_i0_c1_reset_observation.npz',result['durable'])
                self.assertFalse(any(record.get('operation',{}).get('sequence',0)>last for record in result['journal']))

    def test_real_loop_stop_failure_remains_failed_and_preserves_earlier_error(self):
        for fault in (None,'fsync'):
            with self.subTest(fault=fault):
                result=self.exercise(fault=fault,stop_failure=True)
                self.assertIs(result['error'],result['failure'] if fault else result['stop_error'])
                self.assertEqual(result['result']['status'],'failed');self.assertFalse(result['result']['normal_stop'])
                self.assertEqual(sum(r['kind']=='stop_exit' for r in result['journal']),0)
                if fault is None:self.assertEqual(result['position'],696)

if __name__=='__main__':unittest.main()
