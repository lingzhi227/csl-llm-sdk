"""One bounded three-PE simulator run; requires separate exact runtime admission."""
from pathlib import Path
import argparse,hashlib,json,os,time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
from check_multisource import initialized,settled,stable,check_protocol,check_overlap

ROOT=Path(__file__).resolve().parent

def main(case):
    assert case==0 and sorted(os.sched_getaffinity(0))==[0]
    out=ROOT/'cases'/'two-source';out.mkdir(parents=True,exist_ok=False)
    journal=[];captures=[];copies=0;host_bytes=0;launches=0;runner=None
    normal_stop=False;error=None;protocol=None;overlap=None;checks={};baseline=None
    started=time.monotonic()
    def event(kind,**fields):
        assert len(journal)<128
        row=dict(sequence=len(journal),seconds=time.monotonic()-started,kind=kind,**fields)
        with (out/'journal.jsonl').open('ab') as f:f.write((json.dumps(row)+'\n').encode());f.flush();os.fsync(f.fileno())
        journal.append(row)
    def launch(name):
        nonlocal launches
        assert launches<12;event('launch_enter',name=name)
        runner.launch(name,nonblock=False);launches+=1;event('launch_exit',name=name)
    def read(label,name,x,width,count):
        nonlocal copies,host_bytes
        assert copies<14 and 0<=x<3 and 1<=width<=3-x and 1<=count<=128
        size=width*count*4;assert host_bytes+size<=12288
        words=np.zeros(width*count,np.uint32);event('copy_enter',label=label,x=x,width=width,count=count,host_bytes=size)
        runner.memcpy_d2h(words,symbols[name],x,0,width,1,count,**options)
        q=out/(label+'.npz')
        with q.open('xb') as f:np.savez(f,words=words);f.flush();os.fsync(f.fileno())
        fd=os.open(out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        raw=q.read_bytes();assert len(raw)<=4096 and len(captures)<16
        receipt=dict(path=str(q.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),shape=[width,count])
        captures.append(receipt);assert sum(r['bytes'] for r in captures)<=1048576
        copies+=1;host_bytes+=size;event('copy_exit',label=label,receipt=receipt)
        return words.reshape(width,count).tolist()
    try:
        event('create_enter');platform=get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3)
        runner=SdkRuntime(str(ROOT/'out'),platform);event('create_exit')
        symbols={name:runner.get_id(name) for name in ['state','routing','observed','tx_proof','received_buffer']}
        options=dict(data_type=MemcpyDataType.MEMCPY_32BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        event('load_enter');runner.load();event('load_exit');event('run_enter');runner.run();event('run_exit')
        launch('initialize');initial=read('initial-state','state',0,3,48);routing=read('routing','routing',0,3,8)
        initialized(initial,routing);launch('compute')
        for i in range(8):
            launch('snapshot');baseline=read('state-'+str(i),'state',0,3,48)
            if any(row[4] or row[5] or row[18] for row in baseline):break
            if settled(baseline) and all(row[10]>=1 for row in baseline[1:]):break
            if i<7:time.sleep(.025)
        observed=read('observed','observed',0,1,124)
        tx_proof=read('tx-proof','tx_proof',1,2,128)
        rx_banks=read('rx-banks','received_buffer',0,3,31)
        event('final_window_enter');launch('snapshot');final=read('final-state','state',0,3,48);event('final_window_exit')
    except BaseException as exc:error=exc;event('failure',type=type(exc).__name__,message=str(exc)[:256])
    finally:
        if runner is not None:
            event('stop_enter')
            try:runner.stop();normal_stop=True;event('stop_exit')
            except BaseException as exc:
                if error is None:error=exc
                event('stop_error',message=str(exc)[:256])
    if error is None and normal_stop:
        # Report independent scopes even when overlap was not demonstrated.
        for name,fn in [('protocol',lambda:check_protocol(final,observed,tx_proof,rx_banks)),
                        ('stability',lambda:stable(baseline,final)),('overlap',lambda:check_overlap(final))]:
            try:checks[name]=dict(passed=True,summary=fn())
            except BaseException as exc:checks[name]=dict(passed=False,error=str(exc)[:256])
        if not all(v['passed'] for v in checks.values()):error=ValueError('Strict protocol, complete stability and causal overlap all required')
    result=dict(status='passed' if error is None and normal_stop else 'failed',normal_stop=normal_stop,
                captures=captures,copies=copies,host_bytes=host_bytes,launches=launches,H2D=0,
                checks=checks,seconds=time.monotonic()-started,message=None if error is None else str(error)[:256],
                case=case,hardware=False,PEs=3,message_passing_fabric_test=True,original_neural_epochs=0,
                overlap_scope='Issued-to-software-release source/frame lease intervals; no claim of simultaneous DMA or router-cycle arbitration')
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if error:raise error

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);main(p.parse_args().case)
