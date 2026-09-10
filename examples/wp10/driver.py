"""Original BF16 inputs; every candidate arithmetic stage runs on the device."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.attention_numerics import POLICY,fixtures,append,validate,bits,expanded,check_interval
from qwen38.attention_protocol import ARRAYS,READS,INITIAL,LEDGER,NORMAL_EVENTS,OVERFLOW_EVENTS,expected_state,plan,budget,validate_ledger
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
if data!=fixtures():raise ValueError('Original fixture mismatch')
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
    count,native=ARRAYS[name];integer=native==16 or name in ('state','events','control')
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
    return all(back[n][0]==0xa55a and back[n][-1]==0x5aa5 for n in ('cache','input','score_casts','casts')) and all(back[n][0]==1234.5 and back[n][-1]==-4321.25 for n in ('scores','stages'))
def nominal(actual,expected):
    def order(v):
        b=bits(v)>>16
        return 0x8000-(b&0x7fff) if b&0x8000 else 0x8000+b
    return dict(bf16_mismatches=sum(bits(a)!=bits(b) for a,b in zip(actual,expected)),max_bf16_ulp_distance=max(abs(order(a)-order(b)) for a,b in zip(actual,expected)))
def persist():
    for name,value in (('observations',observations),('initializations',initializations),('overflows',overflows)):
        (root/(name+'.json')).write_text(json.dumps(value)+'\n')
cache=data['initial_cache'];generation=0;inits=0;rejects=0;stopped=False
record('load_enter');runner.load();record('load_done')
try:
    record('run_enter');runner.run();record('run_done')
    for name in INITIAL:
        count,native=ARRAYS[name]
        values=guarded16(cache) if name=='cache' else ([0xa55a]+[0]*(count-2)+[0x5aa5] if native==16 else [1234.5]+[0.0]*(count-2)+[-4321.25])
        write(name,values)
    for call,t in enumerate(data['tokens'],1):
        gen,index=t['generation'],t['token'];oracle=oracles[call-1]
        if gen!=generation:
            generation=gen;inits+=1;write('control',[gen,0]);launch('initialize',call)
            st=read('state');physical=read('cache')
            ok=st==expected_state(gen,0,call-1,inits,rejects,True) and physical==guarded16(cache)
            initializations.append(dict(generation=gen,state=st,cache=physical,passed=ok));persist()
            if not ok:raise ValueError('Reset metadata/cache retention failed')
        cache=append(cache,t)
        if cache!=oracle['cache']:raise ValueError('Frozen source cache mismatch')
        inp=guarded16(sum([t[k] for k in ('q','k','v','gate')],[]))
        write('input',inp);write('control',[gen,index]);launch('token',call)
        record('readback_enter',call);back={n:read(n) for n in READS};record('readback_done',call)
        actual=dict(scores=back['scores'][1:-1],score_casts=back['score_casts'][1:-1],stats=back['stats'],stages=back['stages'][1:-1],output_casts=back['casts'][1:-1])
        numerical=validate(t,cache,actual);output=[expanded(v) for v in actual['output_casts'][512:]]
        source=check_interval(output,oracle['source']['output'],oracle['source']['output_bounds'])
        checks=dict(conditional=numerical['passed'],source_output=source['passed'],cache_full_exact=back['cache']==guarded16(cache),
          input_immutable=back['input']==inp,guards=guards(back),state=back['state']==expected_state(gen,index,call,inits,rejects),
          events=back['events']==NORMAL_EVENTS,stable_handles=all(runner.get_id(n)==ids[n] for n in ids))
        observations.append(dict(call=call,generation=gen,token=index,checks=checks,numerical=numerical,source_output=source,
          nominal=nominal(output,oracle['source']['output']),official=nominal(output,oracle['official_output']),actual=actual,readback=back))
        persist();print('TOKEN',call,checks,flush=True)
        if not all(checks.values()):raise ValueError('Attention token validation failed')
        if call==8:
            write('control',[gen,index+1]);launch('token','overflow')
            rejected={n:read(n) for n in READS};rejects+=1
            expected=expected_state(gen,index,call,inits,rejects);expected[5]=4;expected[6]=1
            unchanged={n:rejected[n]==back[n] for n in READS if n not in ('state','events')}
            ok=all(unchanged.values()) and rejected['state']==expected and rejected['events']==OVERFLOW_EVENTS and all(runner.get_id(n)==ids[n] for n in ids)
            overflows.append(dict(passed=ok,unchanged=unchanged,readback=rejected));persist()
            if not ok:raise ValueError('Device overflow mutation/error contract failed')
    if len(traffic)!=budget()['calls'] or launches!=13:raise ValueError('Incomplete schedule')
finally:
    record('stop_enter');runner.stop();stopped=True;record('stop_done')
identity=identity_check();record('python_complete')
(root/'result.json').write_text(json.dumps(dict(passed=True,tokens=10,overflow_calls=1,initializations=2,normal_stop=stopped,identity=identity,budget=budget(),launches=launches,
 scope='synthetic post-QK-normalization/rotation D256 single-query/single-KV head attention; cache8 and raw sigmoid gate; no projections, QK RMS/RoPE, full GQA or full layer'),indent=2)+'\n')
