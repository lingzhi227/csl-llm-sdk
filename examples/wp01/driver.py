"""Four GEMV calls in one SDK runtime; actual outputs and independent CPU oracle."""
from array import array
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import os
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.codecs import encode, decode
from qwen38.numerics import POLICY, weights_from_bits, inputs, reference, evaluate
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform

root=Path(__file__).resolve().parent
policy=json.loads((root/'numerical-policy.json').read_text())
if policy != POLICY: raise ValueError('Frozen numerical policy mismatch')
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Frozen source/input mismatch: '+name)
row_raw=(root/'tile-row-major.bf16').read_bytes()
packed_raw=(root/'tile-column-major.bf16').read_bytes()
weights=weights_from_bits(row_raw)
restored=b''.join(packed_raw[(c*128+r)*2:(c*128+r)*2+2] for r in range(128) for c in range(112))
if restored != row_raw: raise ValueError('Column-major tile mismatch')
observations=[];phases=[]
def record(phase,call=None):
    phases.append({'phase':phase,'call':call,'monotonic':time.monotonic()})
    (root/'lifecycle.json').write_text(json.dumps(phases,indent=2)+'\n')
(root/'runtime-environment.json').write_text(json.dumps({'python':sys.executable,'version':sys.version,
    'numpy':np.__version__,'cwd':str(root),'tmpdir':os.environ.get('TMPDIR'),'pid':os.getpid(),
    'cgroup':Path('/proc/self/cgroup').read_text()},indent=2)+'\n')
record('construct_enter')
runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3))
record('construct_done')
names=('weights_storage','input_storage','output_storage','calls')
ids={name:runner.get_id(name) for name in names}
record('load_enter');runner.load();record('load_done')
record('run_enter');runner.run();record('run_done')
opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
packed_bits=[word[0] for word in struct.iter_unpack('<H',packed_raw)]
weight_frame=encode(array('H',[0xa55a,*packed_bits,0x5aa5]),'memcpy16_containers')
weight_host=np.frombuffer(weight_frame.payload,dtype='<u4').copy()
record('weights_h2d_enter')
runner.memcpy_h2d(ids['weights_storage'],weight_host,0,0,1,1,14338,data_type=MemcpyDataType.MEMCPY_16BIT,**opts)
record('weights_h2d_done')
output_initial=np.array([1234.5]+[17.0]*128+[-4321.25],dtype=np.float32)
runner.memcpy_h2d(ids['output_storage'],output_initial,0,0,1,1,130,data_type=MemcpyDataType.MEMCPY_32BIT,**opts)
for index,(case,vector) in enumerate(inputs()):
    expected,bounds=reference(weights,vector)
    x=np.array([77.5]+vector+[-88.25],dtype=np.float32)
    record('input_h2d_enter',index)
    runner.memcpy_h2d(ids['input_storage'],x,0,0,1,1,114,data_type=MemcpyDataType.MEMCPY_32BIT,**opts)
    record('input_h2d_done',index);record('compute_enter',index)
    runner.launch('compute',nonblock=False)
    record('compute_done',index);record('readback_enter',index)
    y=np.full(130,np.nan,dtype=np.float32);weight_back=np.zeros(14338,dtype=np.uint32)
    x_back=np.zeros(114,dtype=np.float32);counter=np.zeros(1,dtype=np.uint32)
    for name,dest,bits in [('output_storage',y,32),('weights_storage',weight_back,16),('input_storage',x_back,32),('calls',counter,32)]:
        runner.memcpy_d2h(dest,ids[name],0,0,1,1,len(dest),data_type=MemcpyDataType.MEMCPY_16BIT if bits==16 else MemcpyDataType.MEMCPY_32BIT,**opts)
    record('readback_done',index)
    restored_weights=decode(replace(weight_frame,payload=weight_back.astype('<u4',copy=False).tobytes()),'memcpy16_containers')
    numerical=evaluate(y[1:-1].tolist(),expected,bounds,case)
    checks={'weights_and_guards_unchanged':restored_weights==[0xa55a,*packed_bits,0x5aa5],
            'input_and_guards_unchanged':np.array_equal(x.view(np.uint32),x_back.view(np.uint32)),
            'output_guards_unchanged':y[0]==1234.5 and y[-1]==-4321.25,
            'call_counter':int(counter[0])==index+1,
            'export_ids_stable':all(runner.get_id(name)==ids[name] for name in names)}
    entry={'call':index+1,'case':case,'numerical':numerical,'checks':{k:bool(v) for k,v in checks.items()},
           'actual':y[1:-1].tolist(),'expected_f64':expected,'bounds':bounds}
    observations.append(entry)
    (root/f'actual-{index}.f32').write_bytes(y.astype('<f4',copy=False).tobytes())
    (root/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    print('CALL',index+1,case,numerical,checks,flush=True)
record('stop_enter');runner.stop();record('stop_done');record('python_complete')
passed=all(row['numerical']['passed'] and all(row['checks'].values()) for row in observations)
(root/'result.json').write_text(json.dumps({'passed':passed,'calls':4,'runtime_instances':1,
    'scope':'128x112 original-BF16 GEMV, FP32 accumulation, same-runtime warm calls',
    'policy':POLICY,'source_manifest_sha256':hashlib.sha256((root/'source-manifest.json').read_bytes()).hexdigest()},indent=2)+'\n')
if not passed: raise RuntimeError('Numerical/state gate failed; policy unchanged')
