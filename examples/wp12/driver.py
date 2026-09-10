"""Original BF16 inputs; every candidate arithmetic stage runs on the device."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.attention_composed_numerics import POLICY,fixtures,validate_attention,bits,expanded,check_interval
from qwen38.qk_rope_numerics import validate as validate_qk
from qwen38.qk_rope_protocol import PROBES
from qwen38.attention_composed_protocol import ARRAYS,READS,INITIAL,LEDGER,ATTENTION_EVENTS,QK_EVENTS,OVERFLOW_EVENTS,expected_state,plan,budget,validate_ledger
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
root=Path.cwd()
def identity_check():
    counts={}
    for name in ('source-manifest.json','compiled-subset-before-run.json'):
        values=json.loads((root/name).read_text());values=values.get('files',values)
        for path,digest in values.items():
            if hashlib.sha256((root/path).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen identity mismatch: '+path)
        counts[name]=len(values)
    return counts
identity_check();validate_ledger()
if json.loads((root/'numerical-policy.json').read_text())!=POLICY or json.loads((root/'resource-ledger.json').read_text())!=LEDGER:raise ValueError('Policy mismatch')
if not json.loads((root/'sram-admission.json').read_text())['passed']:raise ValueError('SRAM refusal')
data=json.loads((root/'fixtures.json').read_text());oracles=json.loads((root/'oracles.json').read_text())
if {k:data[k] for k in fixtures()}!=fixtures():raise ValueError('Original fixture mismatch')
if [bits(v) for v in data['inverse_frequencies']]!=data['inverse_frequency_bits']:raise ValueError('Static frequency mismatch')
phases=[];traffic=[];observations=[];initializations=[];overflows=[];launches=0

def record(name,call=None):
    phases.append(dict(phase=name,call=call,monotonic=time.monotonic()))
    (root/'lifecycle.json').write_text(json.dumps(phases,indent=2)+'\n')

def journal(row):
    with (root/'transfer-journal.jsonl').open('a') as file:
        file.write(json.dumps({**row,'monotonic':time.monotonic()})+'\n');file.flush();os.fsync(file.fileno())

(root/'runtime-environment.json').write_text(json.dumps(dict(python=sys.executable,version=sys.version,numpy=np.__version__,pid=os.getpid(),cwd=str(root),tmpdir=os.environ.get('TMPDIR'),cgroup=Path('/proc/self/cgroup').read_text()),indent=2)+'\n')
record('construct_enter');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3));record('construct_done')
ids={name:runner.get_id(name) for name in ARRAYS}
opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
def transfer(direction,name,values=None):
    count,native=ARRAYS[name];integer=native==16 or name in ('state','control') or name.endswith('_events')
    dtype=np.uint32 if integer else np.float32
    array=np.zeros(count,dtype=dtype) if values is None else np.array(values,dtype=dtype)
    if array.ndim!=1 or len(array)!=count or array.nbytes>65536:raise ValueError('Physical transfer buffer refusal')
    if len(traffic)>=len(plan()) or (direction,name)!=plan()[len(traffic)]:raise ValueError('Transfer schedule diverged')
    row=dict(sequence=len(traffic)+1,direction=direction,symbol=name,symbol_id=ids[name],pe=0,width=1,count=count,nativebits=native,hostdtype=str(array.dtype),host_bytes=array.nbytes,native_payload_bytes=count*native//8,contiguous=bool(array.flags.c_contiguous))
    journal({**row,'phase':'entry'})
    kwargs={**opts,'data_type':MemcpyDataType.MEMCPY_16BIT if native==16 else MemcpyDataType.MEMCPY_32BIT}
    if direction=='h2d':runner.memcpy_h2d(ids[name],array,0,0,1,1,count,**kwargs)
    else:runner.memcpy_d2h(array,ids[name],0,0,1,1,count,**kwargs)
    journal({**row,'phase':'exit','host_buffer_sha256':hashlib.sha256(array.tobytes()).hexdigest()});traffic.append(row)
    if native==16:array &= np.uint32(65535)
    return array.tolist()
def write(name,values):return transfer('h2d',name,values)
def read(name):return transfer('d2h',name)
def launch(command,call):
    global launches
    record(command+'_enter',call);runner.launch(command,nonblock=False);launches+=1;record(command+'_done',call)
def guarded16(values):return [0xa55a,*[bits(v)>>16 for v in values],0x5aa5]
def guards(back):
    return all(back[n][0]==0xa55a and back[n][-1]==0x5aa5 for n in READS if ARRAYS[n][1]==16) and all(back[n][0]==1234.5 and back[n][-1]==-4321.25 for n in READS if ARRAYS[n][1]==32 and not (n=='state' or n.endswith('_events') or n.endswith('_stats')))
def nominal(actual,expected):
    def order(v):
        b=bits(v)>>16
        return 0x8000-(b&0x7fff) if b&0x8000 else 0x8000+b
    return dict(bf16_mismatches=sum(bits(a)!=bits(b) for a,b in zip(actual,expected)),max_bf16_ulp_distance=max(abs(order(a)-order(b)) for a,b in zip(actual,expected)))
def persist():
    for name,value in (('observations',observations),('initializations',initializations),('overflows',overflows)):
        (root/(name+'.json')).write_text(json.dumps(value)+'\n')
physical_cache=[bits(v)>>16 for v in data['initial_cache']];generation=0;inits=0;rejects=0;stopped=False
weights=guarded16(data['q_weight']+data['k_weight']);frequencies=[1234.5,*data['inverse_frequencies'],-4321.25];probe_inputs=[1234.5,*PROBES,-4321.25]
record('load_enter');runner.load();record('load_done')
try:
    record('run_enter');runner.run();record('run_done')
    for name in INITIAL:
        count,native=ARRAYS[name]
        values={'cache':[0xa55a,*physical_cache,0x5aa5],'attention_input':guarded16([-0.25]*1024),'qk_weights':weights,'qk_frequencies':frequencies,'qk_probe_inputs':probe_inputs}.get(name)
        if values is None:values=[0xa55a]+[0]*(count-2)+[0x5aa5] if native==16 else [1234.5]+[0.0]*(count-2)+[-4321.25]
        write(name,values)
    for call,t in enumerate(data['tokens'],1):
        gen,pos=t['generation'],t['position'];oracle=oracles[call-1]['source']
        if gen!=generation:
            generation=gen;inits+=1;write('control',[gen,0,0]);launch('initialize',call)
            st=read('state');retained=read('cache')
            ok=st==expected_state(gen,0,call-1,inits,rejects,True) and retained==[0xa55a,*physical_cache,0x5aa5]
            initializations.append(dict(generation=gen,state=st,cache=retained,passed=ok));persist()
            if not ok:raise ValueError('Reset metadata/cache retention failed')
        inp=guarded16(t['q']+t['k']+t['v']+t['gate'])
        write('original_input',inp);write('control',[gen,pos,call]);launch('token',call)
        record('readback_enter',call);back={n:read(n) for n in READS};record('readback_done',call)
        qactual={n:back['qk_'+n] if n=='stats' else back['qk_'+n][1:-1] for n in ('stats','rms_stages','normalized','trig','trig_casts','rotary_stages','product_casts','output','probes')}
        qnumerical=validate_qk(t,data['q_weight'],data['k_weight'],data['inverse_frequencies'],qactual,PROBES)
        producer=back['qk_output'][1:-1];consumer=back['attention_input'][1:-1]
        qsource=[check_interval([expanded(v) for v in producer[h*256:(h+1)*256]],oracle['preprocessing']['heads'][h]['output'],oracle['preprocessing']['heads'][h]['bounds']) for h in range(2)]
        physical_cache[pos*256:(pos+1)*256]=producer[256:]
        physical_cache[2048+pos*256:2048+(pos+1)*256]=[bits(v)>>16 for v in t['v']]
        cache_actual=[expanded(v) for v in back['cache'][1:-1]];n=pos+1
        cache_source=check_interval(cache_actual[:n*256],oracle['cache'][:n*256],oracle['cache_bounds'][:n*256])
        derived={**t,**{key:[expanded(v) for v in consumer[j*256:(j+1)*256]] for j,key in enumerate(('q','k','v','gate'))}}
        actual=dict(scores=back['attention_scores'][1:-1],score_casts=back['attention_score_casts'][1:-1],stats=back['attention_stats'],stages=back['attention_stages'][1:-1],output_casts=back['attention_casts'][1:-1])
        numerical=validate_attention(derived,cache_actual,actual);output=[expanded(v) for v in actual['output_casts'][512:]]
        source=check_interval(output,oracle['attention']['output'],oracle['attention']['output_bounds'])
        checks=dict(qk_conditional=qnumerical['passed'],qk_source=all(c['passed'] for c in qsource),attention_conditional=numerical['passed'],final_source=source['passed'],
          device_copy=consumer[:512]==producer and consumer[512:]==inp[513:-1],cache_physical=back['cache']==[0xa55a,*physical_cache,0x5aa5],cache_source=cache_source['passed'],
          input_immutable=back['original_input']==inp,weights_immutable=back['qk_weights']==weights,
          frequencies_immutable=[bits(v) for v in back['qk_frequencies']]==[bits(v) for v in frequencies],probe_inputs_immutable=back['qk_probe_inputs']==probe_inputs,
          guards=guards(back),state=back['state']==expected_state(gen,pos,call,inits,rejects),
          events=back['attention_events']==ATTENTION_EVENTS and back['qk_events']==QK_EVENTS,stable_handles=all(runner.get_id(name)==ids[name] for name in ids))
        observations.append(dict(call=call,generation=gen,position=pos,checks=checks,qk_numerical=qnumerical,qk_source=qsource,attention_numerical=numerical,final_source=source,cache_source=cache_source,
          nominal=nominal(output,oracle['attention']['output']),official=nominal(output,oracles[call-1]['official_output']),
          qk_actual=qactual,attention_actual=actual,readback=back))
        persist();print('TOKEN',call,checks,flush=True)
        if not all(checks.values()):raise ValueError('Composed original-input gate failed')
        if call==8:
            write('control',[gen,pos+1,call+1]);launch('token','overflow')
            rejected={name:read(name) for name in READS};rejects+=1
            expected=expected_state(gen,pos,call,inits,rejects);expected[5]=4;expected[6]=1
            unchanged={name:rejected[name]==back[name] for name in READS if name not in ('state','attention_events')}
            ok=all(unchanged.values()) and rejected['state']==expected and rejected['attention_events']==OVERFLOW_EVENTS and all(runner.get_id(name)==ids[name] for name in ids)
            overflows.append(dict(passed=ok,unchanged=unchanged,readback=rejected));persist()
            if not ok:raise ValueError('Overflow modified preprocessing/cache/stages or counters')
    if len(traffic)!=budget()['calls'] or launches!=13:raise ValueError('Incomplete schedule')
finally:
    record('stop_enter');runner.stop();stopped=True;record('stop_done')
identity=identity_check();record('python_complete')
(root/'result.json').write_text(json.dumps(dict(passed=True,tokens=10,overflow_calls=1,initializations=2,normal_stop=stopped,identity=identity,budget=budget(),launches=launches,
 scope='synthetic original Q/K/V/gate through ordinary RMS256, device partial RoPE64, device operand copy and persistent D256/cache8 attention; no projection/GQA/full layer/long-context or hardware'),indent=2)+'\n')
