"""Connect bounded public-Task uploads to the finite physical copy ledger."""
import math
from weight_upload import Upload,schedule
from upload_monitor import Publisher

def require(ok,message):
    if not ok:raise ValueError(message)

class Session:
    """The IO owner retains this object and its buffers through physical stop."""
    def __init__(self,io,prepared,root,items,profile):
        self.io=io;self.prepared=prepared;self.items=items;self.returned=set()
        self.original=[item['prepared'] for item in items]
        require(len(items)==252 and all(i['symbol']=='weights' for i in items),'Complete weight-only asynchronous upload')
        total=sum(i['host_bytes'] for i in self.original)
        require(io.copies+len(items)<=io.budget['max_copies'] and io.host_bytes+total<=io.budget['max_host_bytes'],
                'Reserve all asynchronous copies within the original finite budget')
        actual={}
        for item,source in zip(items,self.original):
            geometry=tuple(item[k] for k in ('x','y','width','height','count'))
            bound=io.allowed.get(('weights',*geometry))
            require(bound is not None and bound['dtype']=='u32' and 'h2d' in bound['directions'] and
                    bound['host_bytes']==source['host_bytes'],'New compiled binding for every asynchronous ROI')
            actual[geometry]=dict(symbol='weights',dtype='u32',direction='h2d',host_bytes=bound['host_bytes'])
        publisher=Publisher(root,on_progress=lambda kind,index:io.before_call(kind,'weights',None))
        self.upload=Upload(io.runtime,io.types,io.order,self,publisher,profile,actual)
    def event(self,kind,**values):
        # A public nonblocking return is a submitted copy, not completion.
        # The distinct completed event is emitted only after FIFO wait/done.
        if kind=='weight_submit_returned':
            index=values['index'];require(index not in self.returned,'One returned submission per original ROI')
            self.returned.add(index);self.io.copies+=1;self.io.host_bytes+=self.original[index]['host_bytes']
        return self.io.store.event(kind,**values)
    def run(self):
        result=self.upload.run(self.original,lambda index:self.prepared.transfer(self.items[index]))
        require(self.returned==set(range(len(self.items))),'All and only original weight submissions')
        return result

def audit_upload(events,items,prepared):
    """Independent journal ordering, identity, ownership and completion check."""
    originals=[item['prepared'] for item in items];groups=schedule(originals)
    require(len(items)==252 and len(events)==4*len(items)+2*len(groups)+1,'Exact finite asynchronous journal size')
    stream=iter(events);completed=[];total=0;maximum_retained=maximum_tasks=0
    def take(kind):
        value=next(stream,None);require(value is not None and value['kind']==kind,'Exact async event order: '+kind)
        return value
    def duration(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
    for serial,group in enumerate(groups):
        enter=take('weight_batch_enter')
        require(enter['batch']==serial and enter['indices']==group['indices'] and enter['bands']==group['bands'] and
                duration(enter['deadline']),'Exact disjoint-band batch identity')
        retained=0;pins={}
        for index in group['indices']:
            item=originals[index];row=take('weight_submit_enter')
            _,pin=prepared.transfer(items[index]);pins[index]=pin
            retained+=item['host_bytes'];maximum_retained=max(maximum_retained,retained)
            require(row['index']==index and row['batch']==serial and row['rectangle']==[item[k] for k in ('x','y','width','height','count')] and
                    row['host_bytes']==item['host_bytes'] and row['native_sha256']==pin['native_sha256'] and
                    row['retained_bytes']==retained and row['order']=='ROW_MAJOR' and row['nonblock'] is True,
                    'Bounded immutable original ROI submission')
            row=take('weight_submit_returned')
            require(row['index']==index and row['completion_claimed'] is False and duration(row['rpc_seconds']),
                    'Submission return is distinct from completion')
        maximum_tasks=max(maximum_tasks,len(group['indices']))
        require(retained==group['retained_host_bytes']<=64<<20 and len(group['indices'])<=4,'Exact retained buffer bound')
        for index in group['indices']:
            row=take('weight_wait_enter');require(row['index']==index and row['batch']==serial,'Every actual Task waited in FIFO order')
            row=take('weight_task_completed');item=originals[index]
            require(row['index']==index and row['done'] is True and duration(row['wait_seconds']) and
                    row['native_sha256']==pins[index]['native_sha256'] and row['host_bytes']==item['host_bytes'],
                    'Own wait/done completion and immutable value identity before release')
            retained-=item['host_bytes'];total+=item['host_bytes'];completed.append(index)
        row=take('weight_batch_exit')
        require(retained==0 and row['batch']==serial and row['all_owned_tasks_completed'] is True and
                row['seconds']-enter['seconds']<=30,'Batch drained within its nonrenewable window')
    row=take('all_original_weights_completed')
    require(sorted(completed)==list(range(len(items))) and row['copies']==len(items) and row['host_bytes']==total and
            row['batches']==len(groups) and row['initialize_allowed'] is True and next(stream,None) is None,
            'Every original weight completed once before initialize')
    return dict(copies=len(items),host_bytes=total,batches=len(groups),maximum_tasks=maximum_tasks,
                maximum_retained_bytes=maximum_retained,all_tasks_completed=True,server_concurrency_proved=False)
