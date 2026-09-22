"""Bounded27PE simulator capture. All packet progress is on device."""
from pathlib import Path
import hashlib,json,os,time
from schema import SCHEMA,validate

# label, exported symbol, x, y, width, height, u32 words per PE.
FULL=[('state','state',0,0,9,3,32),
      ('input_a_status','input_a_status',0,0,9,3,8),
      ('input_b_status','input_b_status',0,0,9,3,8),
      ('output_status','output_status',0,0,9,3,8),
      ('diagnostic_status','diagnostic_status',0,0,9,2,8),
      ('origin_records','diagnostic_records',0,0,1,2,3072),
      ('peer_records','diagnostic_records',2,0,1,2,1536),
      ('producer_records','diagnostic_records',4,0,3,2,48),
      ('idle1_records','diagnostic_records',1,0,1,2,16),
      ('idle3_records','diagnostic_records',3,0,1,2,16),
      ('idle7_records','diagnostic_records',7,0,2,2,16),
      ('relay_records','relay_records',0,2,9,1,48),
      ('row_packets','row_packets',0,0,1,1,768),
      ('gate_words','gate_words',4,2,1,1,16)]
SPECS={'initial-state':('state',0,0,9,3,32)}
for i in range(8):
    SPECS[f'poll-{i}-state']=('state',0,0,9,3,32)
    SPECS[f'poll-{i}-diag']=('diagnostic_status',0,0,9,2,8)
for cycle in (0,1):
    for name,symbol,x,y,w,h,n in FULL:SPECS[f'final-{cycle}-{name}']=(symbol,x,y,w,h,n)
assert len(SPECS)==45 and sum(w*h*n*4 for symbol,x,y,w,h,n in SPECS.values())==135744
assert sum(w*h*n*4 for name,symbol,x,y,w,h,n in FULL)==50016


def atomic(path,data):
    path=Path(path);raw=(json.dumps(data,indent=2)+'\n').encode()
    if len(raw)>262144:raise ValueError('Metadata bound')
    temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def assemble(records,cycle):
    tiles={f'{x},{y}':{} for y in range(3) for x in range(9)}
    for label,symbol,x,y,w,h,n in FULL:
        raw=records[f'final-{cycle}-{label}']
        if len(raw)!=w*h*n:raise ValueError('Raw flat extent '+label)
        for yy in range(h):
            for xx in range(w):
                tile=tiles[f'{x+xx},{y+yy}'];start=(yy*w+xx)*n
                if symbol in tile:raise ValueError('Duplicate observed export')
                tile[symbol]=raw[start:start+n]
    return dict(schema=SCHEMA,first=4,count=3,qualification_gate=True,tiles=tiles)


def initialized(words):
    if len(words)!=864:raise ValueError('Initial state extent')
    for i in range(27):
        x,y=i%9,i//9;role=5 if y==2 else 4 if y==1 else 1 if x==0 else 2 if x==2 else 3 if 4<=x<=6 else 0
        if words[32*i]!=1 or words[32*i+1]!=role or words[32*i+3]!=x or words[32*i+4]!=0:
            raise ValueError('Initialization state')


def settled(state,diag):
    if any(state[i*32+4] for i in range(27)):return False
    if not all(state[i*32]==2 and state[i*32+2]==1 for i in range(27)):return False
    for y in range(2):
        for x in range(9):
            s=diag[(y*9+x)*8:(y*9+x+1)*8]
            n=52 if x==0 else 81 if x==2 else 3 if 4<=x<=6 else 0
            cap=192 if x==0 else 96 if x==2 else 3 if 4<=x<=6 else 1
            if s!=[1,n,n,0,0,int(y==1 and n>0),cap,1]:return False
    return True


def capture(root,create_runner,MemcpyDataType,MemcpyOrder):
    import numpy as np
    root=Path(root);out=root/'evidence';out.mkdir(parents=True,exist_ok=False)
    if sorted(os.sched_getaffinity(0))!=[0]:raise ValueError('CPU0 required')
    runner=None;events=0;copies=0;host_bytes=0;launches=0;raw_bytes=0
    receipts=[];records={};error=None;normal_stop=False;checks={};started=time.monotonic()
    def event(kind,**fields):
        nonlocal events
        if events>=256:raise ValueError('Journal event bound')
        row=dict(sequence=events,seconds=time.monotonic()-started,kind=kind,**fields)
        with (out/'journal.jsonl').open('ab') as stream:
            stream.write((json.dumps(row)+'\n').encode());stream.flush();os.fsync(stream.fileno())
        if (out/'journal.jsonl').stat().st_size>65536:raise ValueError('Journal byte bound')
        events+=1
    def launch(name):
        nonlocal launches
        if launches>=2 or name not in ('initialize','compute'):raise ValueError('Only two initial launches')
        event('launch_enter',name=name);runner.launch(name,nonblock=False);launches+=1;event('launch_exit',name=name)
    def read(label):
        nonlocal copies,host_bytes,raw_bytes
        if label not in SPECS or label in records or copies>=45:raise ValueError('Capture count/label')
        symbol,x,y,w,h,n=SPECS[label];size=w*h*n*4
        if host_bytes+size>135744:raise ValueError('Capture host-byte bound')
        words=np.zeros(w*h*n,np.uint32)
        event('copy_enter',label=label,bytes=size)
        runner.memcpy_d2h(words,ids[symbol],x,y,w,h,n,**options)
        path=out/(label+'.npz')
        with path.open('xb') as stream:np.savez(stream,words=words);stream.flush();os.fsync(stream.fileno())
        raw=path.read_bytes();raw_bytes+=len(raw)
        if len(raw)>32768 or raw_bytes>196608:raise ValueError('Raw evidence bound')
        receipt=dict(path='evidence/'+path.name,shape=[h,w,n],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        records[label]=words.tolist();receipts.append(receipt);copies+=1;host_bytes+=size
        event('copy_exit',label=label,sha256=receipt['sha256'])
        atomic(root/'capture-progress.json',dict(captures=receipts,copies=copies,host_bytes=host_bytes,launches=launches,journal_events=events))
        return records[label]
    try:
        event('create_enter');runner=create_runner();event('create_exit')
        ids={name:runner.get_id(name) for name in {row[0] for row in SPECS.values()}}
        options=dict(data_type=MemcpyDataType.MEMCPY_32BIT,streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        event('load_enter');runner.load();event('load_exit');event('run_enter');runner.run();event('run_exit')
        launch('initialize');initialized(read('initial-state'));launch('compute')
        for i in range(8):
            state=read(f'poll-{i}-state');diag=read(f'poll-{i}-diag')
            if any(state[j*32+4] for j in range(27)) or settled(state,diag):break
            if i<7:time.sleep(.025)
        for cycle in (0,1):
            for name,*_ in FULL:read(f'final-{cycle}-{name}')
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
        try:
            first=assemble(records,0);second=assemble(records,1)
            if first!=second:raise ValueError('Complete final exports changed between reads')
            checks=dict(first=validate(first),second=validate(second),complete_exports_stable=True)
        except BaseException as exc:error=exc
    result=dict(status='passed' if error is None and normal_stop else 'failed',normal_stop=normal_stop,
                captures=receipts,copies=copies,host_bytes=host_bytes,raw_bytes=raw_bytes,launches=launches,H2D=0,
                checks=checks,seconds=time.monotonic()-started,message=None if error is None else str(error)[:256],
                hardware=False,PEs=27,producers=3,requests=8,fragments=24,halfwords=1024,original_neural_epochs=0)
    atomic(root/'result.json',result)
    if error is not None:raise error
    return result


def validate_saved(root):
    import numpy as np
    root=Path(root);result=json.loads((root/'result.json').read_bytes());records={};total=0;host=0
    if not result['normal_stop'] or result['status']!='passed':raise ValueError('Normal stop and successful capture required')
    for row in result['captures']:
        name=Path(row['path']);label=name.stem
        if name!=Path('evidence')/(label+'.npz') or label not in SPECS or label in records:raise ValueError('Capture path/label')
        path=root/name
        if path.is_symlink() or path.stat().st_size>32768:raise ValueError('Capture regular extent')
        raw=path.read_bytes();total+=len(raw)
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Original capture hash')
        _,x,y,w,h,n=SPECS[label]
        if row['shape']!=[h,w,n]:raise ValueError('Capture shape metadata')
        with np.load(path,allow_pickle=False) as archive:
            words=archive['words']
            if archive.files!=['words'] or words.dtype!=np.uint32 or words.shape!=(h*w*n,):raise ValueError('Raw dtype/shape')
            records[label]=words.tolist();host+=words.nbytes
    polls=sum(k.startswith('poll-') and k.endswith('-state') for k in records)
    expected={'initial-state'}|{f'poll-{i}-{kind}' for i in range(polls) for kind in ('state','diag')}|{f'final-{i}-{name}' for i in (0,1) for name,*_ in FULL}
    if not 1<=polls<=8 or set(records)!=expected:raise ValueError('Exact capture coverage')
    if len(records)!=result['copies'] or result['copies']>45 or host!=result['host_bytes'] or host>135744 or total!=result['raw_bytes'] or total>196608:raise ValueError('Capture accounting')
    if result['launches']!=2 or result['H2D']!=0 or result['hardware'] is not False or result['original_neural_epochs']!=0:raise ValueError('Execution scope')
    initialized(records['initial-state']);first=assemble(records,0);second=assemble(records,1)
    if first!=second:raise ValueError('Saved terminal capture changed')
    validate(first);validate(second)
    return result
