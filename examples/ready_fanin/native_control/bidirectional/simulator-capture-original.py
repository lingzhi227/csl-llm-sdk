"""Finite simulator capture core, separately host-testable without SDK imports."""
from pathlib import Path
import json,os,time
import numpy as np
from capture_store import atomic,save,progress,SPECS
from check_lifecycle import initialized,settled,stable,check_protocol,check_overlap


def capture(root,create_runner,MemcpyDataType,MemcpyOrder):
    root=Path(root)
    if sorted(os.sched_getaffinity(0))!=[0]:raise ValueError('CPU0 required')
    out=root/'evidence';out.mkdir(parents=True,exist_ok=False)
    runner=None;journal=[];receipts=[];copies=0;host_bytes=0;launches=0
    normal_stop=False;error=None;checks={};baseline=None;started=time.monotonic()
    def event(kind,**fields):
        if len(journal)>=128:raise ValueError('Finite journal')
        row=dict(sequence=len(journal),seconds=time.monotonic()-started,kind=kind,**fields)
        with (out/'journal.jsonl').open('ab') as f:
            f.write((json.dumps(row)+'\n').encode());f.flush();os.fsync(f.fileno())
        journal.append(row)
    def launch(name):
        nonlocal launches
        if launches>=11:raise ValueError('Finite launches')
        event('launch_enter',name=name);runner.launch(name,nonblock=False)
        launches+=1;event('launch_exit',name=name)
    def read(label,symbol):
        nonlocal copies,host_bytes
        width,count=SPECS[label];size=width*count*4
        if copies>=17 or width!=3 or count>384 or host_bytes+size>18120:raise ValueError('Finite capture budget')
        value=np.zeros(width*count,np.uint32)
        event('copy_enter',label=label,name=symbol,width=width,count=count,host_bytes=size)
        runner.memcpy_d2h(value,symbols[symbol],0,0,width,1,count,**options)
        receipt=save(root,label,value);receipts.append(receipt);copies+=1;host_bytes+=size
        event('copy_exit',label=label,receipt=receipt)
        progress(root,receipts,dict(copies=copies,launches=launches,host_bytes=host_bytes,journal_events=len(journal)))
        return value.reshape(width,count).tolist()
    try:
        event('create_enter');runner=create_runner();event('create_exit')
        symbols={name:runner.get_id(name) for name in ('state','routing','observed','tx_proof',
                   'received_buffer','transmit_frame','current_source','row_archive')}
        options=dict(data_type=MemcpyDataType.MEMCPY_32BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        event('load_enter');runner.load();event('load_exit')
        event('run_enter');runner.run();event('run_exit')
        launch('initialize');initial=read('initial-state','state');routing=read('routing','routing')
        initialized(initial,routing);launch('compute')
        for i in range(8):
            launch('snapshot');baseline=read('state-'+str(i),'state')
            if any(row[5] or row[18] for row in baseline):break
            if settled(baseline) and all(row[10]>=1 for row in baseline[:2]):break
            if i<7:time.sleep(.025)
        observed=read('observed','observed');proof=read('tx-proof','tx_proof')
        rx=read('rx-banks','received_buffer');frames=read('live-frames','transmit_frame')
        sources=read('live-sources','current_source');rows=read('row-archive','row_archive')
        event('final_window_enter');launch('snapshot');final=read('final-state','state');event('final_window_exit')
    except BaseException as exc:
        error=exc;event('failure',type=type(exc).__name__,message=str(exc)[:256])
    finally:
        if runner is not None:
            event('stop_enter')
            try:runner.stop();normal_stop=True;event('stop_exit')
            except BaseException as exc:
                if error is None:error=exc
                event('stop_error',message=str(exc)[:256])
    if error is None and normal_stop:
        for name,fn in [('protocol',lambda:check_protocol(final,observed,proof,rx,frames,sources,rows)),
                        ('stability',lambda:stable(baseline,final)),('overlap',lambda:check_overlap(final))]:
            try:checks[name]=dict(passed=True,summary=fn())
            except BaseException as exc:checks[name]=dict(passed=False,error=str(exc)[:256])
        if not all(c['passed'] for c in checks.values()):error=ValueError('Protocol, actual row lifecycle, cache and overlap all required')
    result=dict(status='passed' if error is None and normal_stop else 'failed',normal_stop=normal_stop,
                captures=receipts,copies=copies,host_bytes=host_bytes,launches=launches,H2D=0,checks=checks,
                seconds=time.monotonic()-started,message=None if error is None else str(error)[:256],
                hardware=False,PEs=3,packets=9,application_words=200,original_neural_epochs=0)
    atomic(root/'result.json',(json.dumps(result,indent=2)+'\n').encode())
    if error:raise error
    return result
