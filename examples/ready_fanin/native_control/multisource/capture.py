"""Same accepted capture order and checks, using an entered physical context."""
from pathlib import Path
import json,os,time
import numpy as np
from capture_store import save,progress
from hw00.store import atomic_json
from check_multisource import initialized,settled,stable,check_protocol,check_overlap

def capture(root,runner,MemcpyDataType,MemcpyOrder):
    ROOT=Path(root);case=0
    assert case==0 and sorted(os.sched_getaffinity(0))==[0]
    out=ROOT/'evidence';out.mkdir(parents=True,exist_ok=False)
    journal=[];captures=[];copies=0;host_bytes=0;launches=0
    normal_stop=False;error=None;protocol=None;overlap=None;checks={};baseline=None
    started=time.monotonic()
    atomic_json(ROOT/'monitor-deadline.json',dict(active=True,deadline_monotonic=started+90,scope='capture and normal context exit after entry'))
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
        size=width*count*4;assert host_bytes+size<=7748
        words=np.zeros(width*count,np.uint32);event('copy_enter',label=label,x=x,width=width,count=count,host_bytes=size)
        runner.memcpy_d2h(words,symbols[name],x,0,width,1,count,**options)
        receipt=save(ROOT,label,words);captures.append(receipt)
        copies+=1;host_bytes+=size;event('copy_exit',label=label,receipt=receipt)
        progress(ROOT,captures,dict(copies=copies,launches=launches,host_bytes=host_bytes,journal_events=len(journal)))
        return words.reshape(width,count).tolist()
    try:
        event('capture_start')
        symbols={name:runner.get_id(name) for name in ['state','routing','observed','tx_proof','received_buffer']}
        options=dict(data_type=MemcpyDataType.MEMCPY_32BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
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
            try:
                runner.stop();normal_stop=runner.closed
                if not normal_stop:raise ValueError('Actual physical context exit not returned')
                event('stop_exit');atomic_json(ROOT/'monitor-deadline.json',dict(active=False,reason='normal_exit_returned'))
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
                case=case,hardware=True,PEs=3,message_passing_fabric_test=True,original_neural_epochs=0,
                overlap_scope='Issued-to-software-release source/frame lease intervals; no claim of simultaneous DMA or router-cycle arbitration')
    atomic_json(ROOT/'result.json',result)
    if error:raise error
    return result

