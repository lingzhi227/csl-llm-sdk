"""Four full-head token updates; host observes but never updates/reduces state."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.recurrent_numerics import POLICY,step,check
from qwen38.recurrent_protocol import validate_header
from qwen38.composed_protocol import LEDGER,validate_state,validate_ledger
from qwen38.gated_numerics import validate as validate_norm,expanded,bits,bf16_rne,reference as norm_reference
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
root=Path.cwd();manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen file mismatch: '+name)
if json.loads((root/'numerical-policy.json').read_text())!=POLICY or json.loads((root/'resource-ledger.json').read_text())!=LEDGER:raise ValueError('Contract mismatch')
validate_ledger()
if not all(x['passed'] for x in json.loads((root/'sram-admission.json').read_text()).values()):raise ValueError('SRAM refusal')
fixture=json.loads((root/'fixtures.json').read_text());oracles=json.loads((root/'oracles.json').read_text())
norm_cases=json.loads((root/'norm-fixtures.json').read_text())
phases=[];observations=[];init_observations=[]
def record(name,call=None):
    phases.append({'phase':name,'call':call,'monotonic':time.monotonic()})
    (root/'lifecycle.json').write_text(json.dumps(phases,indent=2)+'\n')
(root/'runtime-environment.json').write_text(json.dumps({'python':sys.executable,'version':sys.version,'numpy':np.__version__,
 'pid':os.getpid(),'cwd':str(root),'tmpdir':os.environ.get('TMPDIR'),'cgroup':Path('/proc/self/cgroup').read_text()},indent=2)+'\n')
record('construct_enter');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3));record('construct_done')
names=('recurrent_storage','query_storage','key_storage','value_storage','prediction','final_output','send_packet','receive_packet','control','state','events','parameters','norm_x','norm_w','norm_z','norm_stages','norm_casts','norm_stats')
ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False,data_type=MemcpyDataType.MEMCPY_32BIT)
def h2d(name,pe,values):runner.memcpy_h2d(ids[name],values,pe,0,1,1,len(values),**opts)
def d2h(name,pe,count,dtype=np.float32):
    values=np.zeros(count,dtype=dtype);runner.memcpy_d2h(values,ids[name],pe,0,1,1,count,**opts);return values
def h2d16(name,values):
    opt={**opts,'data_type':MemcpyDataType.MEMCPY_16BIT}
    runner.memcpy_h2d(ids[name],np.array(values,dtype=np.uint32),2,0,1,1,len(values),**opt)
def d2h16(name,count):
    values=np.zeros(count,dtype=np.uint32);opt={**opts,'data_type':MemcpyDataType.MEMCPY_16BIT}
    runner.memcpy_d2h(values,ids[name],2,0,1,1,count,**opt);return (values & 65535).tolist()
def guarded(values):return np.array([1234.5,*values,-4321.25],dtype=np.float32)
record('load_enter');runner.load();record('load_done');record('run_enter');runner.run();record('run_done')
for pe in range(2):
    h2d('recurrent_storage',pe,guarded(fixture['initial_state'][pe*8192:(pe+1)*8192]))
    for name in ('prediction','final_output'):h2d(name,pe,guarded([0.0]*128))
generation=0;initializations=0;previous=None;previous_bounds=None
for call,token in enumerate(fixture['tokens'],1):
    gen,index=token['generation'],token['token']
    if gen!=generation:
        generation=gen;initializations+=1;mode=1 if gen==1 else 0
        previous=fixture['initial_state'] if gen==1 else [0.0]*16384;previous_bounds=[0.0]*16384
        for pe in range(3):h2d('control',pe,np.array([gen,0,mode],dtype=np.uint32))
        record('initialize_enter',call);runner.launch('initialize',nonblock=False);record('initialize_done',call)
        initialized=[d2h('state',pe,12,np.uint32).tolist() for pe in range(3)]
        expected=[gen,0,call-1,initializations,0,0,0,0,0,0,0,0]
        init_observations.append({'generation':gen,'adopt_uploaded_state':bool(mode),'states':initialized,'passed':all(s==expected for s in initialized)})
        if not init_observations[-1]['passed']:raise ValueError('Initialization/reset metadata mismatch')
    ref=step(previous,previous_bounds,token)
    for name in ('prediction','delta','state','output','prediction_bounds','delta_bounds','state_bounds','output_bounds'):
        if ref[name]!=oracles[call-1][name]:raise ValueError('Independent frozen oracle mismatch: '+name)
    record('inputs_enter',call)
    uploaded=[]
    for pe in range(2):
        q=guarded(token['q'][pe*64:(pe+1)*64]);k=guarded(token['k'][pe*64:(pe+1)*64]);v=guarded(token['v']);params=np.array([token['beta'],token['decay']],dtype=np.float32)
        for name,a in (('query_storage',q),('key_storage',k),('value_storage',v),('parameters',params)):h2d(name,pe,a)
        h2d('control',pe,np.array([gen,index,0],dtype=np.uint32));uploaded.append((q,k,v,params))
    norm_case=norm_cases[call-1]
    for n in ('w','z'):h2d16('norm_'+n,[0xa55a,*[bits(v)>>16 for v in norm_case[n]],0x5aa5])
    h2d16('norm_x',[0xa55a]+[0]*128+[0x5aa5]);h2d16('norm_casts',[0xa55a]+[0]*384+[0x5aa5])
    h2d('norm_stages',2,guarded([0.0]*640));h2d('control',2,np.array([gen,index,0],dtype=np.uint32))
    record('inputs_done',call)
    record('consumer_arm_enter',call);runner.launch('arm_consumer',nonblock=False);record('consumer_arm_done',call)
    record('token_enter',call);runner.launch('token',nonblock=False);record('token_done',call)
    record('readback_enter',call)
    state_arrays=[];metadata=[];events=[];received=[];sent=[];preds=[];outputs=[];input_ok=True;guards=True
    for pe in range(2):
        st=d2h('recurrent_storage',pe,8194);state_arrays.extend(st[1:-1].tolist())
        guards=guards and st[0]==1234.5 and st[-1]==-4321.25
        metadata.append(d2h('state',pe,12,np.uint32).tolist());events.append(d2h('events',pe,13,np.uint32).tolist())
        received.append(d2h('receive_packet',pe,131,np.uint32));sent.append(d2h('send_packet',pe,131,np.uint32))
        pred=d2h('prediction',pe,130);y=d2h('final_output',pe,130);preds.append(pred);outputs.append(y)
        guards=guards and pred[0]==y[0]==1234.5 and pred[-1]==y[-1]==-4321.25
        for name,count,original in zip(('query_storage','key_storage','value_storage','parameters'),(66,66,130,2),uploaded[pe]):
            back=d2h(name,pe,count);input_ok=input_ok and np.array_equal(back.view(np.uint32),original.view(np.uint32))
    consumer_packet=d2h('receive_packet',2,131,np.uint32);ack=d2h('send_packet',2,3,np.uint32)
    metadata.append(d2h('state',2,12,np.uint32).tolist());events.append(d2h('events',2,13,np.uint32).tolist())
    nx=d2h16('norm_x',130);nc=d2h16('norm_casts',386);ns=d2h('norm_stages',2,642).tolist()
    norm_observed={'stats':d2h('norm_stats',2,4).tolist()}
    for j,n in enumerate(('norm','gain_pre','exp','gate','precast')):norm_observed[n]=ns[1+128*j:1+128*(j+1)]
    for j,n in enumerate(('norm_bits','gain_bits','output_bits')):norm_observed[n]=nc[1+128*j:1+128*(j+1)]
    norm_input=[expanded(v) for v in nx[1:-1]]
    norm_result=validate_norm({**norm_case,'x':norm_input},norm_observed)
    input_boundary=nx[1:-1]==[bf16_rne(int(v)) for v in consumer_packet[3:]]
    norm_guards=nx[0]==nc[0]==0xa55a and nx[-1]==nc[-1]==0x5aa5 and ns[0]==1234.5 and ns[-1]==-4321.25
    norm_immutable=all(d2h16('norm_'+n,130)==[0xa55a,*[bits(v)>>16 for v in norm_case[n]],0x5aa5] for n in ('w','z'))
    record('readback_done',call)
    if received[0][:3].tolist()!=[gen,index,5]:raise ValueError('ACK header mismatch')
    validate_header(received[1].tolist(),gen,index,2)
    validate_header(sent[0].tolist(),gen,index,4);validate_header(consumer_packet.tolist(),gen,index,4);validate_header(sent[1].tolist(),gen,index,3)
    actual={'prediction':preds[0][1:-1].tolist(),
            'delta':received[1][3:].view(np.float32).tolist(),
            'state':state_arrays,'output':outputs[0][1:-1].tolist()}
    gates={n:check(actual[n],ref[n],ref[n+'_bounds'],token['zero_case']) for n in actual}
    # Phase3 sender body must equal the fully received peer contribution bit-for-bit.
    transport_bits=bool(np.array_equal(sent[1][3:],received[0][3:]))
    norm_transport=bool(np.array_equal(sent[0],consumer_packet) and np.array_equal(consumer_packet[3:],outputs[0][1:-1].view(np.uint32)) and ack.tolist()==[gen,index,5])
    checks={'numerical_stages':all(x['passed'] for x in gates.values()),'protocol_state_events':validate_state(metadata,events,gen,index,call,initializations),
            'gated_norm':norm_result['passed'],'input_boundary_rne':input_boundary,'norm_guards':norm_guards,'norm_inputs_immutable':norm_immutable,'norm_transport_ack':norm_transport,'guards':bool(guards),'immutable_inputs':bool(input_ok),'output_transport_bits':transport_bits,
            'stable_ids':all(runner.get_id(n)==ids[n] for n in names),
            'beta_zero_delta':token['beta']!=0 or all(v==0 for v in actual['delta'])}
    row={'call':call,'generation':gen,'token':index,'stages':gates,'checks':checks,'metadata':metadata,'events':events,
         'receive_headers':[a[:3].tolist() for a in received],'send_buffer_headers':[a[:3].tolist() for a in sent],
         'delta_observation':'peer retained phase2 receive body; root send body now holds global y',
         'norm_observed':norm_observed,'norm_result':norm_result,'norm_input_bf16':nx[1:-1],'consumer_packet':consumer_packet.tolist(),'ack':ack.tolist(),'root_receive_scope':'phase5 ACK header plus retained phase3 output body, checked separately','actual':actual,'expected':{n:ref[n] for n in actual},'bounds':{n:ref[n+'_bounds'] for n in actual}}
    observations.append(row);(root/'observations.json').write_text(json.dumps(observations)+'\n')
    (root/'initializations.json').write_text(json.dumps(init_observations,indent=2)+'\n')
    print('TOKEN',call,gen,index,{n:g['passed'] for n,g in gates.items()},checks,flush=True)
    previous=ref['state'];previous_bounds=ref['state_bounds']
record('stop_enter');runner.stop();record('stop_done');record('python_complete')
passed=all(all(x['checks'].values()) for x in observations)
(root/'result.json').write_text(json.dumps({'passed':passed,'tokens':4,'generations':3,'runtime_instances':1,
 'scope':'synthetic full128x128 recurrent head plus device BF16 boundary and gated RMS128 on third PE; not whole layer',
 'completion':'host waits for all three PE command completions; local commits are not a distributed atomic transaction; error drain/recovery unqualified'},indent=2)+'\n')
if not passed:raise RuntimeError('Recurrence qualification failed')
