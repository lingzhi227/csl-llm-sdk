"""Small async-ledger/deadline/provider fault checks; no SDK or model data."""
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,tempfile,time,weakref
import numpy as np
from runtime_weights import Session,audit_upload
from runtime_monitor import Monitor
from runtime_prepared import Prepared
from upload_monitor import Publisher,atomic

PROFILE=dict(channels=16,maximum_tasks=4,maximum_retained_bytes=64<<20,maximum_copy_bytes=16<<20,batch_seconds=30)
class Journal:
    def __init__(self):self.rows=[];self.started=time.monotonic()
    def event(self,kind,**fields):
        row=dict(kind=kind,sequence=len(self.rows),seconds=time.monotonic()-self.started,**fields);self.rows.append(row);return row
class Provider:
    def transfer(self,item):
        source=item['prepared'];value=np.full((1,1,6144),source['ordinal'],dtype='<u4')
        return value,dict(file=source['file'],sha256='1'*64,shape=list(value.shape),dtype='<u4',native_sha256=hashlib.sha256(value.tobytes()).hexdigest(),selection={})
class Runtime:
    def __init__(self,fail=None):self.tasks=[];self.completed=[];self.fail=fail
    def get_id(self,name):assert name=='weights';return 1
    def memcpy_h2d(self,symbol,buffer,*box,**options):
        index=len(self.tasks);assert options['nonblock'] is True and not buffer.flags.writeable
        if self.fail=='submit' and index==2:raise RuntimeError('Deliberate submission failure')
        parent=self
        class Task:
            def __init__(self):self.source=weakref.ref(buffer);self.finished=False
            def wait(self):
                assert self.source() is not None and not self.source().flags.writeable
                assert parent.completed==list(range(index));parent.completed.append(index);self.finished=True
                return None
            def done(self):return self.finished
        task=Task();self.tasks.append(task);return task
items=[]
for i in range(190):
    y=(2,70,150,230)[(i//20)%4]+i//80
    source=dict(symbol='weights',dtype='u32',order='ROW_MAJOR',x=i%20,y=y,width=1,height=1,count=6144,
                host_shape=[1,1,6144],host_bytes=24576,file=f'tiny-{i}.npy',ordinal=i)
    items.append(dict(symbol='weights',dtype='u32',x=source['x'],y=y,width=1,height=1,count=6144,prepared=source))
def owner(root,fail=None):
    runtime=Runtime(fail)
    allowed={('weights',*[item[k] for k in ('x','y','width','height','count')]):dict(dtype='u32',directions={'h2d'},host_bytes=24576) for item in items}
    io=SimpleNamespace(runtime=runtime,types=SimpleNamespace(MEMCPY_32BIT=32),order=SimpleNamespace(ROW_MAJOR=0),
        copies=338,host_bytes=1547616,budget=dict(max_copies=7289,max_host_bytes=2512945428),allowed=allowed,
        before_call=lambda *args:None,store=Journal())
    io.weight_session=Session(io,Provider(),root,items,PROFILE);return io
cases=[]
with tempfile.TemporaryDirectory() as name:
    root=Path(name);io=owner(root);result=io.weight_session.run()
    checked=audit_upload(io.store.rows,items,Provider())
    assert io.copies==528 and io.host_bytes==1547616+190*24576 and checked['copies']==190
    assert io.runtime.completed==list(range(190)) and not io.weight_session.upload.owned
    cases.append('190 tiny copies integrate exact IO totals and independent FIFO completion journal')
    for mutation in ('native_pin','FIFO_order','premature_barrier'):
        rows=copy.deepcopy(io.store.rows)
        if mutation=='native_pin':next(r for r in rows if r['kind']=='weight_task_completed')['native_sha256']='0'*64
        elif mutation=='FIFO_order':next(r for r in rows if r['kind']=='weight_wait_enter')['index']+=1
        else:rows.insert(3,rows[-1])
        try:audit_upload(rows,items,Provider())
        except ValueError:cases.append(mutation+' rejected by independent journal audit')
        else:raise AssertionError(mutation)
with tempfile.TemporaryDirectory() as name:
    io=owner(Path(name),'submit')
    try:io.weight_session.run()
    except RuntimeError:
        assert io.copies==340 and len(io.weight_session.upload.owned)==3
        assert all(v['buffer'] is not None for v in io.weight_session.upload.owned)
        cases.append('IO owner retains failed upload buffers after exception until physical cleanup')
    else:raise AssertionError('Failure did not propagate')
with tempfile.TemporaryDirectory() as name:
    root=Path(name);clock=[0.]
    profile=dict(runtime_seconds=1080,startup_seconds=420,startup_pre_start_seconds=240,startup_device_seconds=180,capture_seconds=600,normal_stop_seconds=30,
        compute_observer_seconds=90,capture_progress_seconds=15,max_raw_bytes=1<<20,max_raw_files=20,
        max_journal_bytes=1<<20,weight_upload_batches=51)
    atomic(root/'runtime-phase.json',dict(phase='capture',started=0,deadline=600));(root/'journal.jsonl').write_bytes(b'{}\n')
    atomic(root/'startup-segments.json',dict(schema=1,upload_started=0.,device_start_started=0.,device_start_finished=0.))
    def progress(*args):atomic(root/'progress.json',dict(monotonic=clock[0],raw_bytes=0,raw_files=0))
    progress();pub=Publisher(root,clock=lambda:clock[0],on_progress=progress)
    monitor=Monitor(root,profile,0,clock=lambda:clock[0]);pub.begin_batch(0,0,30)
    clock[0]=20;monitor.check(20);pub.end_batch(0);monitor.check(20)
    assert json.loads((root/'progress.json').read_bytes())['monotonic']==20
    cases.append('Integrated parent permits active30s window and sees ordinary progress before retirement')
    pub.begin_batch(1,20,50);clock[0]=51
    try:monitor.check(51)
    except TimeoutError:cases.append('Integrated physical parent rejects blocked weight Task after30s')
    else:raise AssertionError('Extended weight deadline')
    atomic(root/'weight-upload-deadline.json',dict(active=True,kind='weight_upload',batch=2,started=585,deadline=615));clock[0]=601
    try:monitor.check(601)
    except TimeoutError:cases.append('Absolute600s capture still applies during active weight window')
    else:raise AssertionError('Capture extension')
with tempfile.TemporaryDirectory() as name:
    root=Path(name);path=root/'roi.npy';value=np.arange(86*6144,dtype='<u4').reshape(86,1,6144)
    with path.open('xb') as f:np.save(f,value,allow_pickle=False)
    path.chmod(0o444);raw=path.read_bytes()
    pin=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),native_sha256=hashlib.sha256(value.tobytes()).hexdigest(),dtype='<u4',shape=list(value.shape))
    files={f'unused-{i}.npy':pin for i in range(198)};files['roi.npy']=pin
    manifest=dict(kind='derived_vertical_ROIs_from_accepted_original_words',application=[30,1160],all_original_bits_checked=True,
        all_original_words_reused_checked=True,original_weight_bytes=766546368,files=files)
    raw=json.dumps(manifest).encode();(root/'prepared.json').write_bytes(raw)
    provider=Prepared(root,dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    first,_=provider.array('roi.npy');read=provider.disk_bytes_read
    second,_=provider.array('roi.npy');assert isinstance(first,np.memmap) and isinstance(second,np.memmap)
    assert provider.disk_bytes_read==read==pin['bytes'] and provider.cache_bytes==0 and np.array_equal(second[20],value[20])
    path.chmod(0o644)
    try:provider.array('roi.npy')
    except ValueError:cases.append('Large readonly ROI hashed once then mapped; changed identity rejected')
    else:raise AssertionError('Mutable preparation')
report=dict(status='source_integration_fault_checks_passed_no_SDK',cases=cases,count=len(cases),SDK_invocations=0,model_payload_bytes=0,hardware_invocations=0)
(Path(__file__).resolve().parent/'RUNTIME-INTEGRATION-SOURCE-CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
