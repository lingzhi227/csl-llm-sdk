"""Fixed resident qualification loop; external exact admission/guard is required.

This module does not import or construct SDK objects. NativeAdapter wraps the
documented SDK calls supplied by a separately admitted entry point.
"""
import hashlib,io,json,os,time
from pathlib import Path
import numpy as np
from .resident_spatial import TILES,arrays
from .resident_sdk_sequence import sequence,accounting
from .resident_sdk_checks import require,key,check_state,check_generation

def sha(raw):return hashlib.sha256(raw).hexdigest()

class NativeAdapter:
    def __init__(self,runner,data_type,order):
        self.runner=runner;self.data_type=data_type;self.order=order
        self.symbols={name:runner.get_id(name) for name in arrays(TILES[0])}
        require(len(set(self.symbols.values()))==len(self.symbols),'Distinct exported symbols')
    def load(self):return self.runner.load()
    def run(self):return self.runner.run()
    def stop(self):return self.runner.stop()
    def launch(self,name):return self.runner.launch(name,nonblock=False)
    def copy(self,item,value):
        n=item['name'];x,y,w,h,c=(item[k] for k in ('x','y','width','height','count_per_PE'))
        params=dict(data_type=self.data_type.MEMCPY_16BIT if item['bits']==16 else self.data_type.MEMCPY_32BIT,
            streaming=False,order=self.order.ROW_MAJOR,nonblock=False)
        if item['direction']=='h2d':self.runner.memcpy_h2d(self.symbols[n],value,x,y,w,h,c,**params)
        else:self.runner.memcpy_d2h(value,self.symbols[n],x,y,w,h,c,**params)

class Evidence:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir();self.records=0;self.journal_bytes=0;self.file_bytes=0
        self.pending={};self.group=None;self.closed=set();self.flush_started=set();self.started=time.monotonic();self.files=[]
    def event(self,kind,**fields):
        raw=(json.dumps(dict(record=self.records,seconds=time.monotonic()-self.started,kind=kind,**fields),sort_keys=True)+'\n').encode()
        require(self.records<512 and len(raw)<=4096 and self.journal_bytes+len(raw)<=1048576,'Finite journal budget')
        with (self.root/'journal.jsonl').open('ab') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        self.records+=1;self.journal_bytes+=len(raw)
    def observe(self,group,item,value):
        require(group not in self.closed and (self.group is None or self.group==group),'One immutable observation group')
        name=key(item['name'],item['rank']);require(name not in self.pending,'Unique group observation')
        self.group=group;self.pending[name]=value.copy()
        require(sum(a.nbytes for a in self.pending.values())<=327680,'Bounded pending readbacks')
    def flush(self):
        if self.group is None:return
        group=self.group;require(group not in self.flush_started and self.pending,'New nonempty observation group')
        self.flush_started.add(group)
        buffer=io.BytesIO();np.savez(buffer,**self.pending);raw=buffer.getvalue()
        require(len(raw)<=393216 and self.file_bytes+self.journal_bytes+len(raw)<=2097152,'Finite evidence budget')
        name=group+'.npz'
        with (self.root/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        info=dict(path=name,bytes=len(raw),sha256=sha(raw),arrays={k:dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,sha256=sha(a.tobytes())) for k,a in self.pending.items()})
        self.event('group_persisted',group=group,bytes=len(raw),sha256=info['sha256'],array_count=len(self.pending))
        self.files.append(info);self.file_bytes+=len(raw);self.closed.add(group);self.group=None;self.pending={}

def upload(item,weights,hidden):
    require(item['direction']=='h2d' and item['bits']==16,'Only original BF16 uploads')
    value=np.zeros(item['count'],np.uint32);value[0]=0xa55a;value[-1]=0x5aa5
    if item['name']=='weights':
        t=TILES[item['rank']];require(item['generation']==0 and t.matrix,'Initial resident weight upload only')
        value[1:-1]=weights[t.role][:,t.shard*t.columns:(t.shard+1)*t.columns].T.reshape(-1)
    else:
        require(item['name']=='request' and item['rank']==4 and item['generation'] in (1,2,3),'One current stage input')
        value[1:-1]=hidden[item['generation']-1]
    return value

def execute(backend,weights,hidden,oracle,evidence):
    operations=sequence();uploaded={};observed={};reports=[];primary=None;stopped=False
    counts=dict(copies=0,launches=0,H2D=0,D2H=0,host_bytes=0,native_bytes=0)
    next_operation=0
    try:
        evidence.event('load_enter');backend.load();evidence.event('load_exit')
        evidence.event('run_enter');backend.run();evidence.event('run_exit')
        for item in operations:
            require(item['sequence']==next_operation,'Exact next operation')
            g=item['generation'];name=item['name'];started=time.monotonic()
            evidence.event('operation_enter',operation=item)
            if item['kind']=='launch':
                backend.launch(name);counts['launches']+=1
                evidence.event('operation_exit',sequence=next_operation,api_seconds=time.monotonic()-started)
            else:
                dtype=np.float32 if item['dtype']=='float32' else np.uint32
                value=upload(item,weights,hidden) if item['direction']=='h2d' else np.zeros(item['count'],dtype)
                require(value.dtype==dtype and value.shape==(item['count'],) and value.nbytes==item['host_bytes'],'Exact physical host buffer')
                if item['direction']=='h2d':
                    require(evidence.group is None,'Prior evidence must persist before next upload')
                    if name=='weights':
                        require(item['rank'] not in uploaded,'Resident weights loaded once');uploaded[item['rank']]=value.copy()
                    else:require(len(reports)==g-1,'Prior generation verified before reuse')
                backend.copy(item,value)
                evidence.event('operation_exit',sequence=next_operation,api_seconds=time.monotonic()-started,buffer_sha256=sha(value.tobytes()))
                counts['copies']+=1;counts['H2D' if item['direction']=='h2d' else 'D2H']+=1
                counts['host_bytes']+=item['host_bytes'];counts['native_bytes']+=item['native_bytes']
                if item['direction']=='d2h':
                    group='initial' if g==0 else 'weights' if name=='weights' else 'generation'+str(g)
                    evidence.observe(group,item,value);observed[key(name,item['rank'])]=value.copy()
                    if g==0:
                        check_state(value,0);evidence.flush();observed={}
                    elif name=='weights':
                        require(np.array_equal(value&65535,uploaded[item['rank']]),'Original resident weights and guards retained')
                        if item['rank']==7:evidence.flush();observed={}
                    elif name=='request':
                        reports.append(check_generation(g,observed,weights,hidden,oracle))
                        evidence.flush();observed={}
            next_operation+=1
        expected=accounting()
        require(all(counts[k]==expected[k] for k in counts),'Complete transaction accounting')
        require(next_operation==125 and len(reports)==3 and len(uploaded)==6 and len(evidence.closed)==5,'Complete resident graph qualification')
        evidence.event('all_checks_passed')
    except BaseException as error:
        primary=error
        try:evidence.event('failure',error_type=type(error).__name__,message=str(error)[:512],next_operation=next_operation)
        except BaseException:pass
    finally:
        try:
            if evidence.group not in evidence.flush_started:evidence.flush()
        except BaseException as error:
            if primary is None:primary=error
        try:
            evidence.event('stop_enter')
        except BaseException as error:
            if primary is None:primary=error
        try:
            backend.stop();stopped=True
        except BaseException as error:
            if primary is None:primary=error
        try:
            evidence.event('stop_exit',completed=stopped)
        except BaseException as error:
            if primary is None:primary=error
    result=dict(status='passed' if primary is None else 'failed',physical_counts=counts,
        next_operation=next_operation,generations=reports,normal_stop=stopped,
        evidence_files=evidence.files,source_intervals_modified=False,weights_loaded_once=len(uploaded)==6,
        error_type=None if primary is None else type(primary).__name__,message=None if primary is None else str(primary)[:512])
    raw=(json.dumps(result,indent=2)+'\n').encode();require(len(raw)<=131072,'Bounded driver result')
    with (evidence.root.parent/'result.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if primary is not None:raise primary
    return result
