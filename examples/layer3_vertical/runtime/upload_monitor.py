"""Durable nonrenewable weight-batch window for a separate physical parent.

The physical parent calls Parent.check even when its client is blocked inside
memcpy_h2d/Task.wait. True exempts only the ordinary15s capture-idle rule;
the parent must still enforce overall capture/runtime/resource limits.
"""
from pathlib import Path
import json,math,os,time

def atomic(path,value):
 raw=(json.dumps(value,separators=(',',':'))+'\n').encode()
 if len(raw)>4096:raise ValueError('Small weight-monitor record')
 temporary=path.with_suffix(path.suffix+'.tmp')
 with temporary.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(temporary,path);fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def read(path):
 if not path.exists():return None
 if path.is_symlink() or path.stat().st_size>4096:raise ValueError('Small regular monitor file')
 return json.loads(path.read_bytes())
def stamp(value):
 if type(value) not in (int,float) or not math.isfinite(value) or value<0:raise ValueError('Finite monotonic timestamp')
 return value

class Publisher:
 def __init__(self,root,clock=time.monotonic,on_progress=None):
  self.root=Path(root);self.clock=clock;self.active=None;self.last=-1;self.on_progress=on_progress
 def begin_batch(self,serial,started,deadline):
  if self.active is not None or serial!=self.last+1 or deadline-started!=30:raise ValueError('Fresh fixed weight window')
  self.active=(serial,started,deadline);self.last=serial
  atomic(self.root/'weight-upload-deadline.json',dict(active=True,kind='weight_upload',batch=serial,started=started,deadline=deadline))
 def before(self,kind,index):
  if self.active is None or self.clock()>self.active[2]:raise TimeoutError('Weight window before call')
  if self.on_progress is not None:self.on_progress(kind,index)
  atomic(self.root/'weight-upload-progress.json',dict(monotonic=self.clock(),kind=kind,index=index,batch=self.active[0]))
 def end_batch(self,serial):
  if self.active is None or serial!=self.active[0] or self.clock()>self.active[2]:raise TimeoutError('Timely exact weight window completion')
  self.before('weight_batch_completed',-1)
  atomic(self.root/'weight-upload-deadline.json',dict(active=False,completed_batch=serial,completed=self.clock()))
  self.active=None

class Parent:
 def __init__(self,root,started,batch_count,clock=time.monotonic):
  self.root=Path(root);self.started=started;self.batch_count=batch_count;self.clock=clock;self.windows={};self.latest=-1
 def check(self):
  window=read(self.root/'weight-upload-deadline.json');now=self.clock()
  if window is None:return False
  if type(window['active']) is not bool:raise ValueError('Boolean weight window active')
  if not window['active']:return False
  serial=window['batch'];start=stamp(window['started']);deadline=stamp(window['deadline'])
  if type(serial) is not int or not 0<=serial<self.batch_count or serial<self.latest:raise ValueError('Finite monotonic weight batches')
  if window['kind']!='weight_upload' or deadline-start!=30 or not self.started<=start<=now:raise ValueError('Exact admitted weight deadline')
  if serial in self.windows and self.windows[serial]!=(start,deadline):raise ValueError('No weight deadline renewal')
  self.windows[serial]=(start,deadline);self.latest=serial
  if now>deadline:raise TimeoutError('Blocked weight task exceeded absolute30s; physical parent must terminate and release')
  return True
