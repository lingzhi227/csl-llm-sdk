"""One fresh bounded two-PE simulator case. An outer guard admits each finite suite."""
from pathlib import Path
import argparse,hashlib,json,os,time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
from check_case import check,NAMES
from check_snapshot import split_snapshot,check_snapshot

ROOT=Path(__file__).resolve().parent

def main(case):
    assert 0<=case<8 and sorted(os.sched_getaffinity(0))==[0]
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
        assert copies<13 and w in (1,2) and count<=33
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
        symbols={n:runner.get_id(n) for n in ['case_id','state','observed','events','received_buffer','tail_normal_events','consumption_events']}
        options=dict(data_type=MemcpyDataType.MEMCPY_32BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        event('load_enter');runner.load();event('load_exit');event('run_enter');runner.run();event('run_exit')
        values=np.full(2,case,np.uint32);event('h2d_enter')
        runner.memcpy_h2d(symbols['case_id'],values,0,0,2,1,1,**options);h2d=1;event('h2d_exit')
        launch('initialize');launch('exercise')
        for i in range(8):
            launch('snapshot');initial_snapshot=read('state-'+str(i),'state',2,33);state,_,_=split_snapshot(initial_snapshot);rx,tx=state
            settled=(tx[1]==1 and rx[2]==1) if case<4 else (tx[6]==1 and (rx[3]!=0 if case!=7 else rx[18]==3))
            if settled:break
            if i<7:time.sleep(.025)
        observed=read('observed','observed',1,31)[0];received_buffer=read('received-buffer','received_buffer',1,31)[0];events=read('events','events',2,6)
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
            answer=check(case,state,events,observed,received_buffer)
            checked_state,checked_tail,checked_native=check_snapshot(case,initial_snapshot,final_snapshot)
            assert (checked_state,checked_tail,checked_native)==(state,tail_normal_events,consumption_events)
        except BaseException as exc:error=exc
    result=dict(status='passed' if error is None else 'failed',normal_stop=normal_stop,
                captures=captures,copies=copies,launches=launches,H2D=h2d,summary=answer,
                seconds=time.monotonic()-start,message=None if error is None else str(error)[:256],
                case=case,hardware=False,message_passing_fabric_test=False,
                diagnostic_only=True,observation='Complete33word state snapshots; original30 plus tailnormal/native2 trailer',tail_normal_events=tail_normal_events,consumption_events=consumption_events)
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if error:raise error

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True);main(p.parse_args().case)
