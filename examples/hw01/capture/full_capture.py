"""Capture a full resident original layer3 MLP; audit neural math after release."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from full_mlp import Shape,upload_batches

S=Shape(diagnostic_padding=True)
NAMES=('weights','input','partial','result','rounded','gate','up','product','silu','exponential','sigmoid','activation_fp32','product_fp32','output','state')
FLOATS={'input','partial','result','exponential','sigmoid','activation_fp32','product_fp32'}

def require(value,message):
    if not value:raise ValueError(message)

def hash_file(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for raw in iter(lambda:stream.read(1<<20),b''):digest.update(raw)
    return digest.hexdigest()

def load_prepared(root,expected_receipt):
    """Run before constructing SdkRuntime: hash every original packed byte."""
    root=Path(root);path=root/'preparation.json'
    require(hash_file(path)==expected_receipt,'Accepted preparation receipt hash')
    receipt=json.loads(path.read_bytes());packed=root/'packed'
    require(receipt['status']=='passed' and receipt['layer']==3 and receipt['revision']=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0','Original layer3 preparation')
    require(receipt['original_weight_bytes']==534773760 and receipt['native_packed_weight_bytes']==539885568,'All original matrices')
    batches=upload_batches(S);require(len(receipt['transfers'])==len(batches)==86,'Exact upload count')
    for i,(actual,expected) in enumerate(zip(receipt['transfers'],batches)):
        require(all(actual[k]==v for k,v in expected.items()) and actual['file']==f'weights-{i:03d}.bf16','Exact matrix rectangle')
    stamps={}
    for name,spec in receipt['files'].items():
        require(not Path(name).is_absolute() and '/' not in name and '..' not in name,'Safe packed filename')
        p=packed/name;require(p.is_file() and not p.is_symlink() and p.stat().st_size==spec['bytes'],'Packed extent')
        require(hash_file(p)==spec['sha256'],'Prepared payload identity')
        st=p.stat();stamps[name]=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    require(set(receipt['files'])=={f'weights-{i:03d}.bf16' for i in range(86)}|{'hidden.bf16','reference.json','acquisition.json'},'Prepared payload closure')
    hidden=np.fromfile(packed/'hidden.bf16',dtype='<u2').reshape(4,S.hidden)
    return receipt,hidden,stamps

def state_check(value,epoch,prepared=False):
    shape=(S.height,S.width,8);actual=value.reshape(shape);expected=np.zeros(shape,np.uint32)
    expected[:,:,0]=2*epoch-int(prepared);expected[:,:,1]=epoch;expected[:,:,7]=1
    done=epoch-int(prepared)
    for view in (expected[:2*S.channels,:S.projection_shards],expected[S.down_y:S.down_y+S.outputs,:S.down_shards]):
        view[:,:,2:4]=done;view[:,0,5]=3*done
    non=expected[:2*S.channels:2,S.width-2];non[:,4]=done;non[:,5]=3*done;non[:,6]=6*done
    expected[S.down_y,S.width-1,6]=3*S.channels*done
    expected[S.down_y+1,S.width-1,6]=3*S.outputs*done
    if not np.array_equal(actual,expected):
        wrong=np.flatnonzero(actual.reshape(-1)!=expected.reshape(-1))[:4]
        samples=[]
        for index in wrong:
            y,x,word=np.unravel_index(int(index),shape)
            samples.append(dict(x=int(x),y=int(y),word=int(word),expected=int(expected[y,x,word]),actual=int(actual[y,x,word])))
        raise ValueError('All57776PE epoch/compute/packet/release mismatch: '+json.dumps(samples))

def execute(runner,types,order,root,prepared,receipt,hidden,stamps):
    root=Path(root);prepared=Path(prepared)/'packed';evidence=root/'evidence';evidence.mkdir()
    started=time.monotonic();sequence=0;copies=0;launches=0;host_bytes=0;files={};epochs=[]
    stopped=False;primary=None;symbols={};pending={};pending_epoch=0;inflight=None;retention_error=None
    def event(kind,**fields):
        nonlocal sequence
        require(sequence<1024,'Bounded capture journal')
        raw=(json.dumps(dict(sequence=sequence,seconds=time.monotonic()-started,kind=kind,**fields))+'\n').encode()
        with (evidence/'journal.jsonl').open('ab') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        sequence+=1
    def copy(direction,name,x,y,w,h,count,value=None,bits=32):
        nonlocal copies,host_bytes,inflight
        if value is None:value=np.zeros(w*h*count,np.float32 if name in FLOATS else np.uint32)
        require(value.nbytes<=16<<20 and value.size==w*h*count and value.itemsize==4 and value.flags.c_contiguous,'Bounded exact host slots')
        event('copy_enter',direction=direction,name=name,x=x,y=y,width=w,height=h,count=count,bits=bits)
        kwargs=dict(data_type=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT,streaming=False,order=order.ROW_MAJOR,nonblock=False)
        if direction=='d2h':inflight=(dict(name=name,x=x,y=y,width=w,height=h,count=count,bits=bits),value)
        if direction=='h2d':runner.memcpy_h2d(symbols[name],value,x,y,w,h,count,**kwargs)
        else:runner.memcpy_d2h(value,symbols[name],x,y,w,h,count,**kwargs)
        copies+=1;host_bytes+=value.nbytes;event('copy_exit',host_bytes=value.nbytes)
        inflight=None
        return value
    def launch(name):
        nonlocal launches
        event('launch_enter',name=name);runner.launch(name,nonblock=False);launches+=1;event('launch_exit',name=name)
    def stamp(name):
        p=prepared/name;st=p.stat()
        require(not p.is_symlink() and (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==stamps[name],'Prepared file changed since allocation preflight')
        return p
    def save_native(name,value):
        # One rectangle only. Native16 upper host bits carry no payload.
        data=(value&65535).astype('<u2');path=evidence/name
        with path.open('xb') as f:f.write(data.tobytes());f.flush();os.fsync(f.fileno())
        files[name]=dict(bytes=path.stat().st_size,sha256=hash_file(path),dtype='<u2',shape=list(data.shape))
        return files[name]
    try:
        symbols={name:runner.get_id(name) for name in NAMES};event('symbols_resolved')
        launch('initialize');initial=copy('d2h','state',0,0,S.width,S.height,8);pending={'initial_state':initial};state_check(initial,0)
        with (evidence/'initial-state.npy').open('xb') as f:np.save(f,initial)
        files['initial-state.npy']=dict(bytes=(evidence/'initial-state.npy').stat().st_size,
                                       sha256=hash_file(evidence/'initial-state.npy'),shape=list(initial.shape),dtype=str(initial.dtype))
        pending={};del initial
        for b in receipt['transfers']:
            p=stamp(b['file']);value=np.fromfile(p,dtype='<u2').astype(np.uint32)
            copy('h2d','weights',b['x'],b['y'],b['width'],b['height'],12288,value,bits=16);stamp(b['file']);del value
        for epoch in range(1,5):
            pending_epoch=epoch;data={};pending=data
            value=np.zeros(S.projection_shards*96,np.float32)
            value[:S.hidden].view(np.uint32)[:]=hidden[epoch-1].astype(np.uint32)<<16
            copy('h2d','input',S.width-1,0,1,1,len(value),value);del value
            launch('prepare');data['prepared']=copy('d2h','state',0,0,S.width,S.height,8);state_check(data['prepared'],epoch,True)
            launch('compute');data['state']=copy('d2h','state',0,0,S.width,S.height,8);state_check(data['state'],epoch)
            for region,width,height,y in [('projection',S.projection_shards,2*S.channels,0),('down',S.down_shards,S.outputs,S.down_y)]:
                for name,count in [('input',96),('partial',128),('result',128)]:
                    data[region+'_'+name]=copy('d2h',name,0,y,width,height,count)
                data[region+'_rounded']=copy('d2h','rounded',0,y,1,height,128,bits=16)
            for name in ('gate','up','product','silu','exponential','sigmoid','activation_fp32','product_fp32'):
                bits=16 if name in ('gate','up','product','silu') else 32
                data[name]=copy('d2h',name,S.width-2,0,1,2*S.channels,128,bits=bits)
                odd=data[name].reshape(2*S.channels,128)[1::2]
                raw_odd=(odd&65535) if bits==16 else odd.view(np.uint32)
                if np.any(raw_odd):
                    indices=np.argwhere(raw_odd!=0)[:4]
                    raise ValueError('Diagnostic padding '+name+' first nonzero [y,element,word]: '+str([(2*int(y)+1,int(i),int(raw_odd[y,i])) for y,i in indices]))
            data['output']=copy('d2h','output',S.width-1,S.down_y+1,1,1,S.hidden,bits=16)
            path=evidence/f'epoch-{epoch}.npz'
            with path.open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
            files[path.name]=dict(bytes=path.stat().st_size,sha256=hash_file(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in data.items()})
            epochs.append(epoch);pending={};del data;event('epoch_captured',epoch=epoch)
        for b in receipt['transfers']:
            value=copy('d2h','weights',b['x'],b['y'],b['width'],b['height'],12288,bits=16)
            saved=save_native(b['file'],value);del value
            require(saved['sha256']==b['sha256'],'Final resident native weight identity')
        require(copies==253 and launches==9,'Complete253copies/9launches')
        event('capture_checks_passed')
    except BaseException as exc:
        primary=exc
        try:event('failure',message=str(exc)[:1024])
        except BaseException:pass
    finally:
        try:runner.stop();stopped=True
        except BaseException as exc:
            if primary is None:primary=exc
        partial_transfer=None
        if primary is not None and (pending or inflight is not None):
            # Release first. Preserve actual arrays even when a state/transport
            # gate fails before a complete epoch archive can be written.
            failed=dict(pending)
            if inflight is not None:
                partial_transfer,value=inflight;failed['inflight_'+partial_transfer['name']]=value
            try:
                path=evidence/f'failed-epoch-{pending_epoch}.npz'
                with path.open('xb') as f:np.savez(f,**failed);f.flush();os.fsync(f.fileno())
                files[path.name]=dict(bytes=path.stat().st_size,sha256=hash_file(path),incomplete=True,
                                     saved_after_stop_attempt=True,arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in failed.items()})
            except BaseException as exc:retention_error=str(exc)[:1024]
        result=dict(status='captured' if primary is None else 'failed',normal_stop=stopped,
                    complete_epochs=epochs,copies=copies,launches=launches,host_bytes=host_bytes,
                    files=files,seconds=time.monotonic()-started,mathematical_audit_passed=False,
                    full_model=False,message=None if primary is None else str(primary)[:1024],
                    partial_transfer=partial_transfer,failure_retention_error=retention_error)
        with (root/'capture.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    if primary is not None:raise primary
    return result
