"""Four warm gated RMS calls with separately observed BF16 boundaries."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.gated_numerics import POLICY,fixtures,validate,PROBE_BITS,bf16_rne,bits
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
root=Path.cwd()
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen file mismatch: '+name)
if json.loads((root/'numerical-policy.json').read_text())!=POLICY:raise ValueError('Policy mismatch')
if not json.loads((root/'sram-admission.json').read_text())['passed']:raise ValueError('SRAM admission missing')
frozen_fixtures=json.loads((root/'fixtures.json').read_text())
if frozen_fixtures!=fixtures():raise ValueError('Fixture mismatch')
phases=[];observations=[]
def record(name,call=None):
    phases.append({'phase':name,'call':call,'monotonic':time.monotonic()})
    (root/'lifecycle.json').write_text(json.dumps(phases,indent=2)+'\n')
(root/'runtime-environment.json').write_text(json.dumps({'python':sys.executable,'version':sys.version,'numpy':np.__version__,
 'pid':os.getpid(),'cwd':str(root),'tmpdir':os.environ.get('TMPDIR'),'cgroup':Path('/proc/self/cgroup').read_text()},indent=2)+'\n')
record('construct_enter');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3));record('construct_done')
names=('x','w','z','stages','casts','stats','control','state','probes','probe_output')
ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
def h2d(name,values,bits16=False):
    runner.memcpy_h2d(ids[name],values,0,0,1,1,len(values),data_type=MemcpyDataType.MEMCPY_16BIT if bits16 else MemcpyDataType.MEMCPY_32BIT,**opts)
def d2h(name,count,dtype=np.float32,bits16=False):
    values=np.zeros(count,dtype=dtype)
    runner.memcpy_d2h(values,ids[name],0,0,1,1,count,data_type=MemcpyDataType.MEMCPY_16BIT if bits16 else MemcpyDataType.MEMCPY_32BIT,**opts)
    return values & np.uint32(65535) if bits16 else values
record('load_enter');runner.load();record('load_done');record('run_enter');runner.run();record('run_done')
h2d('probes',np.array(PROBE_BITS,dtype=np.uint32))
for call,case in enumerate(frozen_fixtures,1):
    record('inputs_enter',call);inputs={}
    for name in ('x','w','z'):
        inputs[name]=np.array([0xa55a,*[bits(v)>>16 for v in case[name]],0x5aa5],dtype=np.uint32)
        h2d(name,inputs[name],True)
    h2d('stages',np.array([1234.5]+[0.0]*640+[-4321.25],dtype=np.float32))
    h2d('casts',np.array([0xa55a]+[0]*384+[0x5aa5],dtype=np.uint32),True)
    h2d('control',np.array([call],dtype=np.uint32));record('inputs_done',call)
    record('compute_enter',call);runner.launch('compute',nonblock=False);record('compute_done',call)
    record('readback_enter',call)
    stages=d2h('stages',642).tolist();casts=d2h('casts',386,np.uint32,True).tolist()
    state=d2h('state',4,np.uint32).tolist();stats=d2h('stats',4).tolist()
    probe=d2h('probe_output',16,np.uint32,True).tolist()
    immutable=all(np.array_equal(d2h(n,130,np.uint32,True),inputs[n]) for n in inputs)
    immutable=immutable and d2h('probes',16,np.uint32).tolist()==PROBE_BITS
    record('readback_done',call)
    observed={'stats':stats}
    for j,n in enumerate(('norm','gain_pre','exp','gate','precast')):observed[n]=stages[1+128*j:1+128*(j+1)]
    for j,n in enumerate(('norm_bits','gain_bits','output_bits')):observed[n]=casts[1+128*j:1+128*(j+1)]
    numerical=validate(case,observed)
    checks={'numerical':numerical['passed'],'immutable_inputs':bool(immutable),
       'guards':stages[0]==1234.5 and stages[-1]==-4321.25 and casts[0]==0xa55a and casts[-1]==0x5aa5,
       'state':state==[call,call,16*call,0],'probes':probe==[bf16_rne(v) for v in PROBE_BITS],
       'stable_ids':all(runner.get_id(n)==ids[n] for n in names)}
    row={'call':call,'name':case['name'],'observed':observed,'numerical':numerical,'checks':checks,'state':state,'probe_output':probe}
    observations.append(row);(root/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    print('CALL',call,case['name'],checks,numerical['conversions'],flush=True)
record('stop_enter');runner.stop();record('stop_done');record('python_complete')
passed=all(all(r['checks'].values()) for r in observations)
(root/'result.json').write_text(json.dumps({'passed':passed,'calls':4,'runtime_instances':1,
 'scope':'standalone synthetic BF16 direct-gain gated RMS128; not recurrent composition or model layer'},indent=2)+'\n')
if not passed:raise RuntimeError('Gated RMS validation failed')
