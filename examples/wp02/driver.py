"""Two PE GEMV/fabric/SUM/RMS chain; actual intermediate and completion evidence."""
from array import array
from dataclasses import replace
import hashlib,json,math,os,struct,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.codecs import encode,decode
from qwen38.numerics import weights_from_bits,inputs
from qwen38.chain_numerics import POLICY,reference,check,validate
from qwen38.resource_ledger import check as check_ledger
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
root=Path.cwd();manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen input mismatch: '+name)
if json.loads((root/'numerical-policy.json').read_text())!=POLICY:raise ValueError('Policy mismatch')
check_ledger(json.loads((root/'resource-ledger.json').read_text()))
raw=(root/'tile-row-major.bf16').read_bytes();weights=weights_from_bits(raw)
bits=[i[0] for i in struct.iter_unpack('<H',raw)]
shards=[[0xa55a]+[bits[r*112+c] for c in range(pe*56,(pe+1)*56) for r in range(128)]+[0x5aa5] for pe in range(2)]
frame=encode(array('H',shards[0]+shards[1]),'memcpy16_containers')
phases=[];observations=[]
def record(phase,call=None):
    phases.append({'phase':phase,'call':call,'monotonic':time.monotonic()})
    (root/'lifecycle.json').write_text(json.dumps(phases,indent=2)+'\n')
(root/'runtime-environment.json').write_text(json.dumps({'python':sys.executable,'version':sys.version,'numpy':np.__version__,
    'cwd':str(root),'tmpdir':os.environ.get('TMPDIR'),'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text()},indent=2)+'\n')
record('construct_enter');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3));record('construct_done')
names=('weights_storage','input_storage','partial','received','summed','normalized','stats','events','mode')
ids={name:runner.get_id(name) for name in names};opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
def h2d(name,values,count,bits=32):
    runner.memcpy_h2d(ids[name],values,0,0,2,1,count,data_type=MemcpyDataType.MEMCPY_16BIT if bits==16 else MemcpyDataType.MEMCPY_32BIT,**opts)
def d2h(name,count,dtype=np.float32,bits=32):
    values=np.zeros(count*2,dtype=dtype)
    runner.memcpy_d2h(values,ids[name],0,0,2,1,count,data_type=MemcpyDataType.MEMCPY_16BIT if bits==16 else MemcpyDataType.MEMCPY_32BIT,**opts)
    return values.reshape(2,count)
record('load_enter');runner.load();record('load_done');record('run_enter');runner.run();record('run_done')
record('weights_h2d_enter');h2d('weights_storage',np.frombuffer(frame.payload,dtype='<u4').copy(),7170,16);record('weights_h2d_done')
initial=np.array(([1234.5]+[17.0]*128+[-4321.25])*2,dtype=np.float32)
for name in ('partial','received','summed','normalized'):h2d(name,initial,130)
for index,(case,vector) in enumerate(inputs()):
    mode=index%2;expected=reference(weights,vector)
    x=np.array([v for pe in range(2) for v in ([77.5]+vector[pe*56:(pe+1)*56]+[-88.25])],dtype=np.float32)
    record('input_h2d_enter',index);h2d('input_storage',x,58);h2d('mode',np.array([mode,mode],dtype=np.uint32),1);record('input_h2d_done',index)
    record('compute_enter',index);runner.launch('compute',nonblock=False);record('compute_done',index)
    record('readback_enter',index)
    actual={name:d2h(name,130) for name in ('partial','received','summed','normalized')}
    stats=d2h('stats',4);events=d2h('events',14,np.uint32);xback=d2h('input_storage',58)
    wback=d2h('weights_storage',7170,np.uint32,16)
    record('readback_done',index)
    restored=decode(replace(frame,payload=wback.astype('<u4',copy=False).tobytes()),'memcpy16_containers')
    zero=case=='zero_after_nonzero'
    stage={f'partial{pe}':check(actual['partial'][pe,1:-1].tolist(),expected['partial'][pe],expected['partial_bound'][pe],zero) for pe in range(2)}
    stage['summed']=check(actual['summed'][0,1:-1].tolist(),expected['summed'],expected['sum_bound'],zero)
    stage['normalized']=check(actual['normalized'][0,1:-1].tolist(),expected['normalized'],expected['normalized_bound'],zero)
    stage['squares']=check([float(stats[0,0])],[expected['squares']],[expected['squares_bound']])
    stage['q']=check([float(stats[0,1])],[expected['q']],[expected['q_bound']])
    sqroot=math.sqrt(float(stats[0,1])) if stats[0,1]>0 else float('nan')
    sqrt_error=abs(float(stats[0,2])/sqroot-1) if sqroot>0 else float('inf')
    reciprocal_error=abs(float(stats[0,3])*float(stats[0,2])-1)
    checks={'weight_guards_and_bits':restored==shards[0]+shards[1],
            'input_guards_and_bits':np.array_equal(x.view(np.uint32),xback.ravel().view(np.uint32)),
            'all_output_guards':all(np.all(a[:,0]==1234.5) and np.all(a[:,-1]==-4321.25) for a in actual.values()),
            'fabric_bit_exact':np.array_equal(actual['received'][0,1:-1].view(np.uint32),actual['partial'][1,1:-1].view(np.uint32)),
            'device_join_and_unblock':validate(events.astype(object).tolist(),mode,index+1),
            'stable_ids':all(runner.get_id(n)==ids[n] for n in names),
            'sqrt_accuracy':math.isfinite(sqrt_error) and sqrt_error<=POLICY['sqrt_relative_budget'],
            'reciprocal_accuracy':math.isfinite(reciprocal_error) and reciprocal_error<=POLICY['reciprocal_relative_budget']}
    observation={'call':index+1,'case':case,'mode':mode,'stages':stage,'checks':{k:bool(v) for k,v in checks.items()},
                 'sqrt_relative_error':sqrt_error,'reciprocal_relative_error':reciprocal_error,
                 'events':events.tolist(),'stats':stats.tolist(),'actual':{n:a.tolist() for n,a in actual.items()},'reference':expected}
    observations.append(observation);(root/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    print('CALL',index+1,case,'mode',mode,'stages',stage,'checks',checks,flush=True)
record('stop_enter');runner.stop();record('stop_done');record('python_complete')
passed=all(all(s['passed'] for s in o['stages'].values()) and all(o['checks'].values()) for o in observations)
(root/'result.json').write_text(json.dumps({'passed':passed,'runtime_instances':1,'calls':4,
    'scope':'two-PE GEMV/fabric/SUM/RMS128 unit-gain reduced operator chain','policy':POLICY},indent=2)+'\n')
if not passed:raise RuntimeError('WP02 stage/numerical/protocol gate failed')
