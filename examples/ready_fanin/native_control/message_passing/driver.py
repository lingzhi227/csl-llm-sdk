"""One fresh bounded two-PE simulator case. An outer guard admits each finite suite."""
from pathlib import Path
import argparse,hashlib,json,os,time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
from check_consecutive import check,NAMES,split_snapshot,stable_snapshots

ROOT=Path(__file__).resolve().parent

def main(case):
    assert case==0 and sorted(os.sched_getaffinity(0))==[0]
    out=ROOT/'cases'/NAMES[case];out.mkdir(parents=True,exist_ok=False)
    journal=[];captures=[];launches=0;copies=0;h2d=0;normal_stop=False;error=None;answer=None;tail_normal_events=None;consumption_events=None
    start=time.monotonic();runner=None
    def event(kind,**fields):
        assert len(journal)<128
        row=dict(sequence=len(journal),seconds=time.monotonic()-start,kind=kind,**fields)
        with (out/'journal.jsonl').open('ab') as f:f.write((json.dumps(row)+'\n').encode());f.flush();os.fsync(f.fileno())
        journal.append(row)
    def launch(name):
        nonlocal launches
        assert launches<11;event('launch_enter',name=name);runner.launch(name,nonblock=False);launches+=1;event('launch_exit',name=name)
    def read(label,name,w,count):
        nonlocal copies
        assert copies<17 and w in (1,2) and count<=62
        a=np.zeros(w*count,np.uint32);event('copy_enter',label=label,count=count,width=w)
        runner.memcpy_d2h(a,symbols[name],0,0,w,1,count,**options)
        q=out/(label+'.npz')
        with q.open('xb') as f:np.savez(f,words=a);f.flush();os.fsync(f.fileno())
        fd=os.open(out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        raw=q.read_bytes();assert len(raw)<=4096
        receipt=dict(path=str(q.relative_to(ROOT)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),shape=[w,count])
        captures.append(receipt);copies+=1;event('copy_exit',label=label,receipt=receipt)
        return a.reshape(w,count).tolist()
    try:
        event('create_enter')
        platform=get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3)
        runner=SdkRuntime(str(ROOT/'out'),platform);event('create_exit')
        symbols={n:runner.get_id(n) for n in ['case_id','state','observed','events','received_buffer','tail_normal_events','consumption_events','tx_evidence','rx_evidence','transmitted_frame','source_payload']}
        options=dict(data_type=MemcpyDataType.MEMCPY_32BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        event('load_enter');runner.load();event('load_exit');event('run_enter');runner.run();event('run_exit')
        values=np.full(2,case,np.uint32);event('h2d_enter')
        runner.memcpy_h2d(symbols['case_id'],values,0,0,2,1,1,**options);h2d=1;event('h2d_exit')
        launch('initialize');launch('exercise')
        for i in range(8):
            launch('snapshot');initial_snapshot=read('state-'+str(i),'state',2,33);state,_,_=split_snapshot(initial_snapshot);rx,tx=state
            settled=tx[1]==2 and rx[2]==2 and initial_snapshot[0][30:33]==[0,2,0] and initial_snapshot[1][30:33]==[0,0,0]
            if settled:break
            if i<7:time.sleep(.025)
        observed=read('observed','observed',2,62);received_buffer=read('received-buffer','received_buffer',2,31);events=read('events','events',2,6)
        tx_evidence=read('tx-evidence','tx_evidence',2,24);rx_evidence=read('rx-evidence','rx_evidence',2,12)
        transmitted_frame=read('transmitted-frame','transmitted_frame',2,32);source_payload=read('source-payload','source_payload',2,31)
        # Add an explicit later observation window after the first settled poll
        # and payload reads. This is finite evidence, not proof that no callback
        # could ever remain pending beyond this window.
        event('final_window_enter')
        launch('snapshot');final_snapshot=read('final-state','state',2,33)
        state,tail_normal_events,consumption_events=split_snapshot(final_snapshot)
        final_events=read('final-events','events',2,6)
        assert final_events==events,'Callbacks changed during final observation window'
        events=final_events;event('final_window_exit')
    except BaseException as exc:error=exc;event('failure',type=type(exc).__name__,message=str(exc)[:256])
    finally:
        if runner is not None:
            event('stop_enter')
            try:runner.stop();normal_stop=True;event('stop_exit')
            except BaseException as exc:
                if error is None:error=exc
                event('stop_error',message=str(exc)[:256])
    # Validate saved pre-stop observations only after a successful normal stop.
    if error is None:
        try:
            if not settled:raise ValueError('No complete settled snapshot baseline within eight polls')
            assert stable_snapshots(initial_snapshot,final_snapshot)==(state,tail_normal_events,consumption_events)
            answer=check(state,events,observed,received_buffer,tx_evidence,rx_evidence,transmitted_frame,source_payload,tail_normal_events,consumption_events)
        except BaseException as exc:error=exc
    result=dict(status='passed' if error is None else 'failed',normal_stop=normal_stop,
                captures=captures,copies=copies,launches=launches,H2D=h2d,summary=answer,
                seconds=time.monotonic()-start,message=None if error is None else str(error)[:256],
                case=case,hardware=False,message_passing_fabric_test=True,
                diagnostic_only=True,tail_normal_events=tail_normal_events,consumption_events=consumption_events)
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if error:raise error

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);main(p.parse_args().case)
