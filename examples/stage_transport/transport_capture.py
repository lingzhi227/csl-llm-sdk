"""Capture the small final-topology transport fixture; no original model I/O."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

NAMES=('descriptor','weights','packed_weights','input','partial','result','output','counters','frame_counters','state')
FLOATS={'input','partial','result'}


def require(test,message):
    if not test:raise ValueError(message)


def weight_words(stripe):
    """Exact diagnostic coefficients, in accepted column-major local order."""
    count=stripe['columns'];weights=np.zeros((count,96,128),np.uint16)
    for ordinal in range(count):
        valid=min(96,stripe['input_words']-96*ordinal)
        for row in range(stripe['rows']):weights[ordinal,row%valid,row]=0x3f80
    return weights


def descriptors(plan):
    values=np.zeros((plan['height'],plan['width'],6),np.uint32)
    for s in plan['stripes']:
        x,y,c=s['x'],s['y'],s['columns']
        for ordinal in range(c):
            values[y,x+ordinal]=[s['phase'],s['group'],x+c,y,s['rows'],min(96,s['input_words']-96*ordinal)]
        values[y,x+c]=[s['phase'],s['group'],0,0,s['rows'],0]
    return values


def capture(runner,types,order,root):
    root=Path(root);plan=json.loads((root/'transport-plan.json').read_bytes())
    require(plan['epochs']==2 and plan['matrix_PEs']==408,'Exact connected fixture')
    evidence=root/'evidence';evidence.mkdir();started=time.monotonic();sequence=0
    copies=0;launches=0;files={};completed=[];pending={};stopped=False;primary=None
    symbols={}
    def event(kind,**details):
        nonlocal sequence
        require(sequence<1024,'Journal bound')
        with (evidence/'journal.jsonl').open('ab') as f:
            f.write((json.dumps(dict(sequence=sequence,kind=kind,seconds=time.monotonic()-started,**details))+'\n').encode());f.flush();os.fsync(f.fileno())
        sequence+=1
    def copy(direction,name,x,y,w,h,n,value=None,bits=32):
        nonlocal copies
        if value is None:value=np.zeros(w*h*n,np.float32 if name in FLOATS else np.uint32)
        value=np.ascontiguousarray(value.reshape(-1))
        require(value.itemsize==4 and value.size==w*h*n and value.nbytes<=16<<20,'Transfer extent/bound')
        event('copy_enter',direction=direction,name=name,x=x,y=y,width=w,height=h,count=n,bits=bits)
        opts=dict(data_type=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT,
                  streaming=False,order=order.ROW_MAJOR,nonblock=False)
        if direction=='h2d':runner.memcpy_h2d(symbols[name],value,x,y,w,h,n,**opts)
        else:
            pending['inflight_'+name]=value
            runner.memcpy_d2h(value,symbols[name],x,y,w,h,n,**opts)
            del pending['inflight_'+name]
        copies+=1;event('copy_exit');return value
    def launch(name):
        nonlocal launches
        event('launch_enter',name=name);runner.launch(name,nonblock=False);launches+=1;event('launch_exit',name=name)
    def archive(name,arrays):
        path=evidence/name
        with path.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
        require(path.stat().st_size<=16<<20,'Archive bound')
        files[name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def state_gate(value,epoch,prepared=False):
        data=value.reshape(plan['height'],plan['width'],8);expected=np.zeros_like(data)
        expected[:,:,0]=2*epoch-int(prepared);expected[:,:,1]=epoch
        prior=epoch-int(prepared);expected[0,0,2]=4*prior;expected[0,0,3]=5*prior
        require(np.array_equal(data,expected),'All-PE command barrier/state')
    try:
        symbols={name:runner.get_id(name) for name in NAMES}
        launch('initialize')
        initial=copy('d2h','state',0,0,200,3,8);state_gate(initial,0);archive('initial.npz',dict(state=initial))
        expected_descriptors=descriptors(plan)
        copy('h2d','descriptor',0,0,200,3,6,expected_descriptors,bits=16)
        for s in plan['stripes']:
            native=weight_words(s).reshape(-1).astype(np.uint32)
            packed=native[0::2]|(native[1::2]<<16)
            copy('h2d','packed_weights',s['x'],s['y'],s['columns'],1,6144,packed,bits=32)
            observed=copy('d2h','weights',s['x'],s['y'],s['columns'],1,12288,bits=16)&65535
            require(np.array_equal(observed,native),'Packed32 upload and legacy16 alias agree before arithmetic')
        for epoch in (1,2):
            pending={};launch('prepare')
            pending['prepared']=copy('d2h','state',0,0,200,3,8);state_gate(pending['prepared'],epoch,True)
            launch('compute');pending['state']=copy('d2h','state',0,0,200,3,8);state_gate(pending['state'],epoch)
            for s in plan['stripes']:
                tag=str(s['group']);x,y,c=s['x'],s['y'],s['columns']
                for name,n in [('input',96),('partial',128),('result',128),('counters',4)]:
                    pending[tag+'_'+name]=copy('d2h',name,x,y,c,1,n)
                pending[tag+'_root']=copy('d2h','output',x,y,1,1,128,bits=16)
                pending[tag+'_consumer']=copy('d2h','output',x+c,y,1,1,128,bits=16)
            pending['origin']=copy('d2h','frame_counters',0,0,1,1,5)
            archive('epoch-'+str(epoch)+'.npz',pending);pending={};completed.append(epoch)
        for s in plan['stripes']:
            native=(copy('d2h','weights',s['x'],s['y'],s['columns'],1,12288,bits=16)&65535).astype(np.uint16)
            pending={'weights':native};archive('retained-'+str(s['group'])+'.npz',pending)
            require(np.array_equal(native,weight_words(s).reshape(-1)),'All diagnostic resident weights retained');pending={}
            packed=copy('d2h','packed_weights',s['x'],s['y'],s['columns'],1,6144,bits=32)
            expected=native[0::2].astype(np.uint32)|(native[1::2].astype(np.uint32)<<16)
            archive('retained-packed-'+str(s['group'])+'.npz',dict(packed=packed))
            require(np.array_equal(packed,expected),'Both weight aliases retain the same original halfwords')
        final=(copy('d2h','descriptor',0,0,200,3,6,bits=16)&65535).reshape(3,200,6)
        archive('descriptors.npz',dict(descriptor=final));require(np.array_equal(final,expected_descriptors),'Runtime descriptors retained')
    except BaseException as error:primary=error
    finally:
        try:runner.stop();stopped=True
        except BaseException as error:
            if primary is None:primary=error
        if primary is not None and pending:
            try:archive('failed-partial.npz',pending)
            except BaseException as error:files['failure_retention_error']=str(error)
        result=dict(status='captured' if primary is None else 'failed',normal_stop=stopped,
            complete_epochs=completed,copies=copies,launches=launches,files=files,
            seconds=time.monotonic()-started,error=None if primary is None else str(primary),
            original_model=False,transport_audit_passed=False)
        with (root/'capture.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if primary is not None:raise primary
    return result
