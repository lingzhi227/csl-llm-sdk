"""Four warm RMS calls; exact device conversion is distinct from oracle parity."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.rms_numerics import N,POLICY,PROBE_BITS,fixtures,reference,close,gain_gate,bits,from_bits,f32,bf16_rne,bf16_of_real
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
root=Path.cwd()
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen file mismatch: '+name)
if json.loads((root/'numerical-policy.json').read_text())!=POLICY:raise ValueError('Policy mismatch')
if not json.loads((root/'sram-admission.json').read_text())['passed']:raise ValueError('SRAM admission missing')
frozen_fixtures=json.loads((root/'fixtures.json').read_text())
if frozen_fixtures!=[[name,x,w] for name,x,w in fixtures()]:raise ValueError('Fixture mismatch')
phases=[];observations=[]
def record(name,call=None):
    phases.append({'phase':name,'call':call,'monotonic':time.monotonic()})
    (root/'lifecycle.json').write_text(json.dumps(phases,indent=2)+'\n')
(root/'runtime-environment.json').write_text(json.dumps({'python':sys.executable,'version':sys.version,'numpy':np.__version__,
 'pid':os.getpid(),'cwd':str(root),'tmpdir':os.environ.get('TMPDIR'),'cgroup':Path('/proc/self/cgroup').read_text()},indent=2)+'\n')
record('construct_enter');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3));record('construct_done')
names=('staging','gain_output_storage','stats','control','state','probe_input','probe_output','timing')
ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
def h2d(name,values,bits16=False):
    runner.memcpy_h2d(ids[name],values,0,0,1,1,len(values),data_type=MemcpyDataType.MEMCPY_16BIT if bits16 else MemcpyDataType.MEMCPY_32BIT,**opts)
def d2h(name,count,dtype=np.float32,bits16=False):
    values=np.zeros(count,dtype=dtype)
    runner.memcpy_d2h(values,ids[name],0,0,1,1,count,data_type=MemcpyDataType.MEMCPY_16BIT if bits16 else MemcpyDataType.MEMCPY_32BIT,**opts)
    return values & np.uint32(65535) if bits16 else values
record('load_enter');runner.load();record('load_done');record('run_enter');runner.run();record('run_done')
probe=np.array([0x12345678,*PROBE_BITS,0x87654321],dtype=np.uint32)
h2d('probe_input',probe);h2d('probe_output',np.array([0xa55a]+[0x7777]*16+[0x5aa5],dtype=np.uint32),True)
for call,(case,x,w) in enumerate(frozen_fixtures,1):
    ref=reference(x,w);gain_bits=[bits(g)>>16 for g in w]
    record('h2d_enter',call)
    h2d('staging',np.array([1234.5,*x,-4321.25],dtype=np.float32))
    h2d('gain_output_storage',np.array([0xa55a,*gain_bits,0x5aa5],dtype=np.uint32),True)
    h2d('control',np.array([call],dtype=np.uint32));record('h2d_done',call)
    record('compute_enter',call);runner.launch('compute',nonblock=False);record('compute_done',call)
    runner.launch('convert_probe',nonblock=False);record('readback_enter',call)
    staging=d2h('staging',N+2);actual=staging[1:-1].tolist()
    output=d2h('gain_output_storage',N+2,np.uint32,True).tolist()
    scalars=d2h('stats',4).tolist();state=d2h('state',7,np.uint32).tolist()
    probe_back=d2h('probe_input',18,np.uint32);probe_result=d2h('probe_output',18,np.uint32,True).tolist()
    timing=d2h('timing',6,np.uint32,True).tolist();record('readback_done',call)
    squares,q,sqrt_q,inverse=scalars
    if not all(math.isfinite(v) for v in scalars) or min(q,sqrt_q,inverse)<=0:raise ValueError('Invalid scalars')
    sqrt_error=abs(sqrt_q-math.sqrt(q))/math.sqrt(q)
    inverse_error=abs(inverse-1/sqrt_q)/(1/sqrt_q)
    stages={'squares':close([squares],[ref['squares']],[ref['squares_bound']]),
            'denominator':close([q],[ref['q']],[ref['q_bound']]),
            'precast':close(actual,ref['expected'],ref['bounds']),
            'gain_using_actual_inverse':gain_gate(x,w,inverse,actual)}
    expected_bits=[bf16_rne(bits(a)) for a in actual]
    oracle_bits=[bf16_of_real(a) for a in ref['expected']]
    alt_before_gain=[bf16_rne(bits(f32(from_bits(bf16_rne(bits(f32(a*inverse)))<<16)*f32(1+g)))) for a,g in zip(x,w)]
    rounding_order_differences=sum(a!=b for a,b in zip(expected_bits,alt_before_gain))
    start=timing[0]+(timing[1]<<16)+(timing[2]<<32);end=timing[3]+(timing[4]<<16)+(timing[5]<<32)
    checks={'stage_gates':all(v['passed'] for v in stages.values()),
            'sqrt_actual_denominator':sqrt_error<=POLICY['sqrt_relative_budget'],
            'inverse_actual_sqrt':inverse_error<=POLICY['reciprocal_relative_budget'],
            'device_rne_of_actual_precast':output[1:-1]==expected_bits,
            'conversion_probe':probe_result==[0xa55a,*[bf16_rne(b) for b in PROBE_BITS],0x5aa5],
            'probe_input_immutable':np.array_equal(probe_back,probe),
            'staging_guards':staging[0]==1234.5 and staging[-1]==-4321.25,
            'output_guards':output[0]==0xa55a and output[-1]==0x5aa5,
            'state':state==[call,call,2,0,N,call,N],
            'stable_ids':all(runner.get_id(n)==ids[n] for n in names),
            'kernel_timestamp':0<((end-start)&((1<<48)-1))<(1<<40),
            'zero_or_annihilated_elements':all(y==0 for a,g,y in zip(x,w,actual) if a==0 or g==-1),
            'rounding_order_adversary':case!='rounding_boundary' or rounding_order_differences>0}
    row={'call':call,'case':case,'stages':stages,'checks':{k:bool(v) for k,v in checks.items()},'state':state,
         'scalar_actual':scalars,'sqrt_relative_error':sqrt_error,'inverse_relative_error':inverse_error,
         'oracle_bf16_mismatch_count':sum(a!=b for a,b in zip(output[1:-1],oracle_bits)),
         'device_gain_lifecycle':'consumed then overwritten by BF16 output; host original gain and frozen fixture immutable','oracle_bf16_parity_is_not_conversion_gate':True,'gain_before_cast_adversary_differences':rounding_order_differences,
         'actual_precast':actual,'oracle_precast':ref['expected'],'precast_bounds':ref['bounds'],
         'device_bf16':output[1:-1],'actual_precast_rne':expected_bits,'oracle_bf16':oracle_bits,
         'probe_input_bits':PROBE_BITS,'probe_output_bits':probe_result[1:-1],
         'raw_timestamp_words':timing,'kernel_cycles':(end-start)&((1<<48)-1)}
    observations.append(row);(root/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    print('CALL',call,case,{k:v['passed'] for k,v in stages.items()},row['checks'],'oracle_bf16_mismatches',row['oracle_bf16_mismatch_count'],flush=True)
record('stop_enter');runner.stop();record('stop_done');record('python_complete')
passed=all(all(x['checks'].values()) for x in observations)
(root/'result.json').write_text(json.dumps({'passed':passed,'calls':4,'runtime_instances':1,'scope':'synthetic full5120 ordinary RMS and actual BF16 RNE; not model layer','policy':POLICY},indent=2)+'\n')
if not passed:raise RuntimeError('RMS qualification gate failed')
