"""Eight causal preprocessing tokens with persistent BF16 history and exact stage casts."""
import hashlib,json,math,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.preprocess_numerics import POLICY,fixtures,history_step,validate,PROBE_BITS,bf16_rne,bits,expanded,convolution
from qwen38.preprocess_protocol import LEDGER,validate_ledger
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
root=Path.cwd()
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen file mismatch: '+name)
if json.loads((root/'numerical-policy.json').read_text())!=POLICY:raise ValueError('Policy mismatch')
validate_ledger()
if json.loads((root/'resource-ledger.json').read_text())!=LEDGER:raise ValueError('Ledger mismatch')
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
names=('history','weights','input','parameters','stages','casts','outputs','norm_stats','gates','beta_bits','control','state','probes','probe_output')
ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
def h2d(name,values,bits16=False):
    runner.memcpy_h2d(ids[name],values,0,0,1,1,len(values),data_type=MemcpyDataType.MEMCPY_16BIT if bits16 else MemcpyDataType.MEMCPY_32BIT,**opts)
def d2h(name,count,dtype=np.float32,bits16=False):
    values=np.zeros(count,dtype=dtype)
    runner.memcpy_d2h(values,ids[name],0,0,1,1,count,data_type=MemcpyDataType.MEMCPY_16BIT if bits16 else MemcpyDataType.MEMCPY_32BIT,**opts)
    return values & np.uint32(65535) if bits16 else values
record('load_enter');runner.load();record('load_done');record('run_enter');runner.run();record('run_done')
oracles=json.loads((root/'oracles.json').read_text());data=frozen_fixtures
weight_bits=np.array([0xa55a,*[bits(x)>>16 for x in data['weights']],0x5aa5],dtype=np.uint32)
h2d('weights',weight_bits,True);h2d('history',np.array([0xa55a]+[0]*1536+[0x5aa5],dtype=np.uint32),True)
h2d('stages',np.array([1234.5]+[0.0]*1152+[-4321.25],dtype=np.float32))
h2d('outputs',np.array([1234.5]+[0.0]*512+[-4321.25],dtype=np.float32))
h2d('casts',np.array([0xa55a]+[0]*768+[0x5aa5],dtype=np.uint32),True)
h2d('probes',np.array(PROBE_BITS,dtype=np.uint32))
generation=0;initializations=0;history=[0.0]*1536;initialization_records=[]
for call,token in enumerate(data['tokens'],1):
    gen,index=token['generation'],token['token']
    if gen!=generation:
        generation=gen;initializations+=1;history=[0.0]*1536
        h2d('control',np.array([gen,0],dtype=np.uint32))
        record('initialize_enter',call);runner.launch('initialize',nonblock=False);record('initialize_done',call)
        hs=d2h('history',1538,np.uint32,True).tolist();st=d2h('state',8,np.uint32).tolist()
        expected=[gen,0,call-1,initializations,0,0,0,0]
        ok=hs==[0xa55a]+[0]*1536+[0x5aa5] and st==expected
        initialization_records.append({'generation':gen,'state':st,'zero_history_and_guards':ok})
        if not ok:raise ValueError('Reset failed')
    history=history_step(history,token['x'])
    if history!=oracles[call-1]['history']:raise ValueError('Frozen history mismatch')
    record('inputs_enter',call)
    inp=np.array([0xa55a,*[bits(x)>>16 for x in token['x']],0x5aa5],dtype=np.uint32)
    params=np.array([0xa55a,*[bits(token[n])>>16 for n in ('a','b','A_log','dt_bias')],0x5aa5],dtype=np.uint32)
    h2d('input',inp,True);h2d('parameters',params,True);h2d('control',np.array([gen,index],dtype=np.uint32));record('inputs_done',call)
    record('token_enter',call);runner.launch('token',nonblock=False);record('token_done',call)
    vector_state=d2h('state',8,np.uint32).tolist()
    if vector_state!=[gen,index-1,call-1,initializations,min(index,4),2,0,index-1]:raise ValueError('Vector phase counter mismatch')
    if not np.array_equal(d2h('parameters',6,np.uint32,True),params):raise ValueError('Parameters changed between commands')
    record('finalize_enter',call);runner.launch('finalize',nonblock=False);record('finalize_done',call);record('readback_enter',call)
    stage=d2h('stages',1154).tolist();cast=d2h('casts',770,np.uint32,True).tolist();out=d2h('outputs',514).tolist()
    ns=d2h('norm_stats',8).tolist();gates=d2h('gates',15).tolist();beta=d2h('beta_bits',1,np.uint32,True).tolist()[0]
    hs=d2h('history',1538,np.uint32,True).tolist();st=d2h('state',8,np.uint32).tolist()
    immutable=np.array_equal(d2h('weights',1538,np.uint32,True),weight_bits) and np.array_equal(d2h('input',386,np.uint32,True),inp) and np.array_equal(d2h('parameters',6,np.uint32,True),params)
    probe=d2h('probe_output',16,np.uint32,True).tolist();immutable=immutable and d2h('probes',16,np.uint32).tolist()==PROBE_BITS
    record('readback_done',call)
    observed={'conv':stage[1:385],'activation':stage[385:769],'activation_exp':stage[769:1153],
      'conv_bits':cast[1:385],'activation_bits':cast[385:769],'q_normalized':out[1:129],'k':out[129:257],
      'q':out[257:385],'v':out[385:513],'q_stats':ns[:4],'k_stats':ns[4:],'gates':gates,'beta_bits':beta}
    numerical=validate(token,data['weights'],history,observed)
    guards=stage[0]==out[0]==1234.5 and stage[-1]==out[-1]==-4321.25 and cast[0]==hs[0]==0xa55a and cast[-1]==hs[-1]==0x5aa5
    checks={'numerical':numerical['passed'],'history_exact':hs[1:-1]==[bits(x)>>16 for x in history],
      'state':st==[gen,index,call,initializations,min(index,4),3,0,index],'guards':guards,
      'immutable_inputs_weights':bool(immutable),'conversion_probes':probe==[bf16_rne(v) for v in PROBE_BITS],
      'stable_ids':all(runner.get_id(n)==ids[n] for n in names)}
    conv_expected,conv_bounds=convolution(history,data['weights'])
    row={'call':call,'generation':gen,'token':index,'checks':checks,'state':st,'vector_state':vector_state,'numerical':numerical,'observed':observed,
      'history':history,'actual_history_bits':hs[1:-1],'conv_expected':conv_expected,'conv_bounds':conv_bounds,'probe_output':probe}
    observations.append(row);(root/'observations.json').write_text(json.dumps(observations)+'\n')
    (root/'initializations.json').write_text(json.dumps(initialization_records,indent=2)+'\n')
    print('TOKEN',call,gen,index,checks,flush=True)
record('stop_enter');runner.stop();record('stop_done');record('python_complete')
passed=all(all(row['checks'].values()) for row in observations)
(root/'result.json').write_text(json.dumps({'passed':passed,'tokens':8,'generations':2,'runtime_instances':1,
 'scope':'synthetic selected128Q/128K/128V preprocessing and width4 history; not projections, all10240 channels, recurrence composition or layer'},indent=2)+'\n')
if not passed:raise RuntimeError('Preprocessing failed')
