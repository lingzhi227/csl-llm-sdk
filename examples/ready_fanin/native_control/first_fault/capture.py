"""Preserve run004 traffic and sink reads; add two bounded row0 fault-bank reads."""
import json,os,time
from pathlib import Path
import numpy as np
from check_capture import check,EXPECTED
from check_fault_capture import inspect_faults
from capture_store import save,progress
from hw00.store import atomic_json

def capture(root,runner,DataType,Order):
    root=Path(root);evidence=root/'evidence';evidence.mkdir();started=time.monotonic();sequence=0;receipts=[];copies=0;launches=0;host_bytes=0
    stopped=False;error=None;summary=None;partial=None;fault_report=None
    atomic_json(root/'monitor-deadline.json',dict(active=True,deadline_monotonic=started+90,scope='capture and normal context exit after context entry'))
    def event(kind,**fields):
        nonlocal sequence
        if sequence>=96:raise ValueError('Finite journal')
        raw=(json.dumps(dict(sequence=sequence,seconds=time.monotonic()-started,kind=kind,**fields))+'\n').encode()
        with (evidence/'journal.jsonl').open('ab') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        sequence+=1
    opts=dict(data_type=DataType.MEMCPY_32BIT,streaming=False,order=Order.ROW_MAJOR,nonblock=False)
    def copy(label,name,x,y,w,h,count):
        nonlocal copies,host_bytes
        if copies>=15 or count>3072 or w*h*count>178*128:raise ValueError('Finite rectangle')
        value=np.zeros(w*h*count,np.uint32)
        event('copy_enter',label=label,name=name,box=[x,y,w,h],count=count,host_bytes=int(value.nbytes))
        runner.memcpy_d2h(value,symbols[name],x,y,w,h,count,**opts)
        receipt=save(root,label,value);receipts.append(receipt);copies+=1;host_bytes+=int(value.nbytes)
        if host_bytes>283776:raise ValueError('Aggregate raw host bound')
        event('copy_exit',label=label,receipt=receipt)
        progress(root,receipts,dict(copies=copies,launches=launches,host_bytes=host_bytes,journal_events=sequence))
        return value.reshape(w*h,count).tolist()
    def launch(name):
        nonlocal launches
        event('launch_enter',name=name);runner.launch(name,nonblock=False);launches+=1;event('launch_exit',name=name)
    try:
        event('capture_start');symbols={n:runner.get_id(n) for n in ['state','diagnostic_status','diagnostic_records','fault_snapshot']}
        launch('initialize');initial=copy('initial','state',0,0,178,2,4)
        for y in range(2):
            for x in range(178):
                role=4 if y==1 else 1 if x==0 else 2 if x==2 else 3 if 40<=x<176 else 0
                if initial[y*178+x]!=[1,role,0,x]:raise ValueError('Actual initialized state')
        launch('compute')
        for i in range(8):
            status=copy(f'progress-{i:02d}','diagnostic_status',0,1,178,1,8)
            if any(status[x][3] or status[x][4] for x in EXPECTED):break
            if all(status[x][1]>=n for x,n in EXPECTED.items()):break
            if i<7:time.sleep(.025)
        origin=copy('origin','diagnostic_records',0,1,1,1,3072)[0]
        peer=copy('peer','diagnostic_records',2,1,1,1,1536)[0]
        producers=copy('producers','diagnostic_records',40,1,136,1,48)
        status=copy('final-status','diagnostic_status',0,1,178,1,8)
        fault_first=copy('first-fault','fault_snapshot',0,0,178,1,128)
        fault_final=copy('first-fault-final','fault_snapshot',0,0,178,1,128)
        fault_report=inspect_faults(fault_first,fault_final)
        atomic_json(root/'fault-report.json',fault_report)
        partial=check(status,origin,peer,producers,require_complete=False)
        atomic_json(root/'partial-summary.json',partial)
        summary=check(status,origin,peer,producers)
        if not fault_report['capture_stable'] or fault_report['latched_faults'] or fault_report['incomplete_columns']:
            raise ValueError('Unchanged strict650 checks plus stable all-zero fault banks required')
    except BaseException as exc:error=exc;event('failure',error=type(exc).__name__,message=str(exc)[:512])
    finally:
        event('stop_enter')
        try:
            runner.stop();stopped=runner.closed
            if not stopped:raise ValueError('Actual context exit not returned')
            event('stop_exit');atomic_json(root/'monitor-deadline.json',dict(active=False,reason='normal_exit_returned'))
        except BaseException as exc:
            if error is None:error=exc
            event('stop_error',message=str(exc)[:512])
        result=dict(status='passed' if error is None and stopped else 'failed',normal_stop=stopped,copies=copies,launches=launches,host_slot_bytes=host_bytes,journal_events=sequence,captures=receipts,summary=summary,partial_summary=partial,fault_report=fault_report,hardware=True,neural_math=False,rounds=1,reset=False,seconds=time.monotonic()-started,message=None if error is None else str(error)[:512])
        atomic_json(root/'result.json',result)
    if error:raise error
    return result
