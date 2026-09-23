"""Qualified deadline publisher and atomic metadata writer, reused for the chain."""
from pathlib import Path
import json,os,time
from runtime_boundary import require
from runtime_liveness import operation, mark

def atomic_json(path,value,*,durable=True):
    path=Path(path);raw=(json.dumps(value,separators=(',',':'))+'\n').encode()
    require(len(raw)<=16<<20,'Bounded capture metadata')
    temporary=path.with_suffix(path.suffix+'.tmp')
    with operation('metadata_write',path.name):
        with temporary.open('wb') as f:
            f.write(raw);f.flush()
            if durable:
                with operation('metadata_file_fsync',path.name):os.fsync(f.fileno())
        with operation('metadata_replace',path.name):os.replace(temporary,path)
        if durable:
            fd=os.open(path.parent,os.O_RDONLY)
            try:
                with operation('metadata_directory_fsync',path.name):os.fsync(fd)
            finally:os.close(fd)

def advisory_json(path,value):
    """Atomic visibility only; never evidence of durable completion."""
    atomic_json(path,value,durable=False)

class Timing:
    """Publish hard deadlines BEFORE calls; the separate parent enforces them."""
    def __init__(self,root,profile,store):
        self.root=Path(root);self.profile=profile;self.store=store;self.started=time.monotonic();self.compute_deadline=None
    def before(self,kind,name,rectangle=None):
        now=time.monotonic()
        require(now<self.started+self.profile['capture_seconds'],'Capture wall deadline')
        if self.compute_deadline is not None:require(now<self.compute_deadline,'Compute/observer absolute deadline')
        mark('before_'+kind,name,scope='capture')
        advisory_json(self.root/'progress.json',dict(monotonic=now,operation=[kind,name],rectangle=rectangle,
            raw_bytes=self.store.raw_bytes,raw_files=len(self.store.files)))
    def begin_compute(self,serial):
        require(self.compute_deadline is None,'No outstanding serial')
        started=time.monotonic();self.compute_deadline=started+self.profile['compute_observer_seconds']
        atomic_json(self.root/'monitor-deadline.json',dict(active=True,kind='compute_observer',serial=serial,
            started=started,deadline=self.compute_deadline))
    def observed(self):
        require(self.compute_deadline is not None and time.monotonic()<self.compute_deadline,'Timely durable observer')
        self.compute_deadline=None
        # Refresh progress before retiring the long window so the parent
        # never applies15s to an already completed long observer RPC.
        self.before('boundary','observer-complete')
        atomic_json(self.root/'monitor-deadline.json',dict(active=False,reason='complete_origin_observer_durable'))
