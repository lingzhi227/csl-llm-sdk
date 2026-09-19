"""Finite SDK capture reads only independently drained row1 diagnostic sinks."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
from check_capture import check,EXPECTED
ROOT=Path(__file__).resolve().parent;START=time.monotonic();SEQ=0

def event(kind,**fields):
    global SEQ
    assert SEQ<256
    raw=(json.dumps(dict(sequence=SEQ,seconds=time.monotonic()-START,kind=kind,**fields))+'\n').encode()
    with (ROOT/'journal.jsonl').open('ab') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    SEQ+=1

def main():
    assert sorted(os.sched_getaffinity(0))==[0]
    event('runtime_create_enter');platform=get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3)
    runner=SdkRuntime('out',platform);event('runtime_create_exit');receipts=[];copies=0;launches=0;error=None;stopped=False;summary=None;partial=None
    symbols={n:runner.get_id(n) for n in ['state','diagnostic_status','diagnostic_records']}
    options=dict(data_type=MemcpyDataType.MEMCPY_32BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
    def copy(label,name,x,y,w,h,count):
        nonlocal copies
        assert copies<37 and count<=3072 and w*h*count<=136*48
        value=np.zeros(w*h*count,np.uint32)
        event('copy_enter',label=label,name=name,box=[x,y,w,h],count=count,host_bytes=value.nbytes)
        runner.memcpy_d2h(value,symbols[name],x,y,w,h,count,**options)
        # Return is durably saved before any subsequent SDK call.
        path=ROOT/'evidence'/f'{copies:02d}-{label}.npz';path.parent.mkdir(exist_ok=True)
        with path.open('xb') as f:np.savez(f,words=value);f.flush();os.fsync(f.fileno())
        raw=path.read_bytes();assert len(raw)<=131072
        receipts.append(dict(path=str(path.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),shape=[w*h,count]))
        copies+=1;event('copy_exit',label=label,receipt=receipts[-1]);return value.reshape(w*h,count).tolist()
    try:
        event('load_enter');runner.load();event('load_exit');event('run_enter');runner.run();event('run_exit')
        event('launch_enter',name='initialize');runner.launch('initialize',nonblock=False);launches+=1;event('launch_exit',name='initialize')
        initial=copy('initial','state',0,0,178,2,4)
        for y in range(2):
            for x in range(178):
                role=4 if y==1 else 1 if x==0 else 2 if x==2 else 3 if 40<=x<176 else 0
                assert initial[y*178+x]==[1,role,0,x]
        event('launch_enter',name='compute');runner.launch('compute',nonblock=False);launches+=1;event('launch_exit',name='compute')
        for i in range(32):
            status=copy(f'progress-{i:02d}','diagnostic_status',0,1,178,1,8)
            if any(status[x][3] or status[x][4] for x in EXPECTED):break
            if all(status[x][1]>=count for x,count in EXPECTED.items()):break
            # Avoid a long fresh-transfer idle interval; no unbounded poll or retry.
            if i<31:time.sleep(.025)
        origin=copy('origin','diagnostic_records',0,1,1,1,192*16)[0]
        peer=copy('peer','diagnostic_records',2,1,1,1,96*16)[0]
        producers=copy('producers','diagnostic_records',40,1,136,1,3*16)
        status=copy('final-status','diagnostic_status',0,1,178,1,8)
        partial=check(status,origin,peer,producers,require_complete=False)
        (ROOT/'partial-summary.json').write_text(json.dumps(partial,indent=2)+'\n')
        summary=check(status,origin,peer,producers)
    except BaseException as exc:error=exc;event('failure',error=type(exc).__name__,message=str(exc)[:512])
    finally:
        event('stop_enter')
        try:runner.stop();stopped=True;event('stop_exit')
        except BaseException as exc:
            if error is None:error=exc
            event('stop_error',message=str(exc)[:512])
        result=dict(status='passed' if error is None else 'failed',normal_stop=stopped,copies=copies,launches=launches,raw_files=receipts,summary=summary,partial_summary=partial,hardware=False,neural_math=False,seconds=time.monotonic()-START,message=None if error is None else str(error)[:512])
        with (ROOT/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if error:raise error
if __name__=='__main__':main()
