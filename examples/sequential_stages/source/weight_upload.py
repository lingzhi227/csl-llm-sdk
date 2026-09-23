"""Bounded original-weight async upload using installed2.10 public Task methods.

No runtime is allocated here. The admitted caller owns the physical parent
watchdog, normal stop and cancellation/reap on any failure. Configuration and
state remain blocking. This optimization changes host transport only.
"""
import hashlib,time
BANDS=[(0,65)]+[(65+i*73,65+(i+1)*73) for i in range(15)]
MAX_TASKS=4;MAX_RETAINED=64<<20;MAX_COPY=16<<20

def bands(item):
 y=item['y'];end=y+item['height']
 if not 0<=y<end<=1160:raise ValueError('Actual installed channel height')
 return {i for i,(a,b) in enumerate(BANDS) if max(a,y)<min(b,end)}

def schedule(items):
 """Deterministic first eligible group; each window has disjoint DMA ybands."""
 remaining=list(range(len(items)));result=[]
 while remaining:
  batch=[];used=set();total=0
  for index in remaining:
   item=items[index];size=item['host_bytes'];channels=bands(item)
   if not 0<size<=MAX_COPY:raise ValueError('Bounded host ROI')
   if not channels&used and len(batch)<MAX_TASKS and total+size<=MAX_RETAINED:
    batch.append(index);used|=channels;total+=size
  if not batch:raise ValueError('Finite weight batch must make progress')
  result.append(dict(indices=batch,bands=sorted(used),retained_host_bytes=total))
  chosen=set(batch);remaining=[i for i in remaining if i not in chosen]
 return result

class Upload:
 """Retain every source until its own FIFO Task.wait() and Task.done().

    On failure, failed outstanding records remain strongly referenced in this
    object through physical owner cleanup. There is no retry or buffer reuse.
    The parent monitors the fixed batch deadline even if an SDK call blocks.
 """
 def __init__(self,runtime,types,order,store,monitor,profile,actual_bindings,clock=time.monotonic):
  self.runtime=runtime;self.types=types;self.order=order;self.store=store;self.monitor=monitor
  self.clock=clock;self.profile=profile;self.bindings=actual_bindings;self.owned=[];self.used=False;self.failed=False
  self.completed=[];self.maximum_retained=0;self.maximum_tasks=0;self.deadline=None
  if profile!=dict(channels=16,maximum_tasks=4,maximum_retained_bytes=64<<20,maximum_copy_bytes=16<<20,batch_seconds=30):
   raise ValueError('Exact separately admitted weight upload profile')
 def event(self,kind,**values):return self.store.event(kind,**values)
 def before(self,kind,index):
  if self.deadline is None or self.clock()>self.deadline:raise TimeoutError('Nonrenewable weight upload batch deadline')
  self.monitor.before(kind,index)
 def run(self,items,load_pinned_roi):
  import numpy as np
  if self.used:raise ValueError('Exactly one original weight upload pass; no retry')
  self.used=True;groups=schedule(items);total=0
  # Static actual-bank bindings must cover every proposed weight ROI exactly.
  for index,item in enumerate(items):
   key=tuple(item[k] for k in ('x','y','width','height','count'))
   if item['symbol']!='weights' or item['dtype']!='u32' or item['order']!='ROW_MAJOR' or item['count']!=6144:
    raise ValueError('Original native u32 matrix bank only')
   if item['host_shape']!=[item['height'],item['width'],item['count']]:raise ValueError('Physical ROW_MAJOR host axes')
   if self.bindings.get(key)!=dict(symbol='weights',dtype='u32',direction='h2d',host_bytes=item['host_bytes']):
    raise ValueError('Every async ROI requires its new actual compiled binding')
  symbol=self.runtime.get_id('weights')
  try:
   for serial,group in enumerate(groups):
    if self.owned:raise ValueError('Previous FIFO window must be drained')
    started=self.clock();self.deadline=started+self.profile['batch_seconds']
    self.monitor.begin_batch(serial,started,self.deadline)
    self.event('weight_batch_enter',batch=serial,indices=group['indices'],bands=group['bands'],deadline=self.deadline)
    for index in group['indices']:
     item=items[index];source,pin=load_pinned_roi(index)
     if source.dtype!=np.dtype('<u4') or list(source.shape)!=item['host_shape'] or not source.flags.c_contiguous:
      raise ValueError('Exact ROW_MAJOR prepared u32 source')
     if source.nbytes!=item['host_bytes'] or hashlib.sha256(source.tobytes()).hexdigest()!=pin['native_sha256']:
      raise ValueError('Pinned original ROI identity')
     # Own one immutable independent buffer, even if the loader reuses scratch.
     buffer=np.array(source,copy=True,order='C');buffer.setflags(write=False);del source
     owned=dict(index=index,buffer=buffer,task=None,pin=pin);self.owned.append(owned)
     retained=sum(v['buffer'].nbytes for v in self.owned)
     if retained>MAX_RETAINED or len(self.owned)>MAX_TASKS:raise ValueError('Async ownership bound')
     self.maximum_retained=max(self.maximum_retained,retained);self.maximum_tasks=max(self.maximum_tasks,len(self.owned))
     self.before('weight_submit',index);self.event('weight_submit_enter',index=index,batch=serial,
       rectangle=[item[k] for k in ('x','y','width','height','count')],host_bytes=buffer.nbytes,
       native_sha256=pin['native_sha256'],order='ROW_MAJOR',nonblock=True,retained_bytes=retained)
     submitted=self.clock()
     task=self.runtime.memcpy_h2d(symbol,buffer,*[item[k] for k in ('x','y','width','height','count')],
       data_type=self.types.MEMCPY_32BIT,order=self.order.ROW_MAJOR,streaming=False,nonblock=True)
     if not callable(getattr(task,'wait',None)) or not callable(getattr(task,'done',None)):
      raise ValueError('Installed public Task interface required')
     owned['task']=task;self.event('weight_submit_returned',index=index,rpc_seconds=self.clock()-submitted,completion_claimed=False)
     del buffer
    # Installed H2D shortcut uses last_finished: FIFO avoids waiting a later
    # task first and bypassing removal of earlier pending entries.
    while self.owned:
     owned=self.owned[0];index=owned['index'];task=owned['task']
     self.before('weight_wait',index);self.event('weight_wait_enter',index=index,batch=serial)
     started_wait=self.clock();task.wait() # Normal installed return is None.
     self.before('weight_done',index)
     if task.done() is not True:raise RuntimeError('Task wait did not establish public completed status')
     self.event('weight_task_completed',index=index,wait_seconds=self.clock()-started_wait,done=True,
       native_sha256=owned['pin']['native_sha256'],host_bytes=owned['buffer'].nbytes)
     total+=owned['buffer'].nbytes;self.completed.append(index);self.owned.pop(0)
     del owned,task
    self.monitor.end_batch(serial);self.deadline=None;self.event('weight_batch_exit',batch=serial,all_owned_tasks_completed=True)
   if sorted(self.completed)!=list(range(len(items))) or total!=sum(i['host_bytes'] for i in items):
    raise ValueError('Exactly every original weight ROI completed')
   self.event('all_original_weights_completed',copies=len(items),host_bytes=total,batches=len(groups),initialize_allowed=True)
   return dict(copies=len(items),host_bytes=total,batches=len(groups),maximum_tasks=self.maximum_tasks,
      maximum_retained_bytes=self.maximum_retained,channels=16,server_concurrency_proved=False,measured_speedup=False)
  except BaseException as exc:
   self.failed=True
   self.event('weight_upload_failed',error=type(exc).__name__,message=str(exc),completed=self.completed,
      outstanding=[dict(index=v['index'],task_returned=v['task'] is not None,retained_bytes=v['buffer'].nbytes) for v in self.owned],
      buffers_retained_until_physical_cleanup=True,initialize_allowed=False)
   raise
