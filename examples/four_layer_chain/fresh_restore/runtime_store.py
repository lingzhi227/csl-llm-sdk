"""Bounded source-only raw capture utility for a future complete Layer0 runtime.

No SDK dependency or execution. The caller owns actual runtime admission,
resource supervision and stop/reap. Every raw file has an immutable identity;
partial journals remain evidence if a later operation fails.
"""
from pathlib import Path
import hashlib
import json
import os
import time
from runtime_liveness import operation, mark
from runtime_timing import atomic_json, advisory_json

COMMIT_EVENTS = 128
COMMIT_FENCES = {'runtime_enter','runtime_ready','phase','weight_batch_exit',
                 'all_original_weights_completed','all_restore_uploads_returned',
                 'all_layer_state_passed','transport_fence_passed','stop_enter','stop_exit'}


class Store:
    def __init__(self, root, *, max_raw_bytes, max_journal_bytes, max_files, max_events):
        self.root=Path(root)
        self.raw=self.root/'raw';self.raw.mkdir(exist_ok=False)
        self.journal=self.root/'journal.jsonl'
        self.fd=os.open(self.journal,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        self.started=time.monotonic()
        self.max_raw_bytes=max_raw_bytes;self.max_journal_bytes=max_journal_bytes
        self.max_files=max_files;self.max_events=max_events
        self.raw_bytes=self.journal_bytes=self.sequence=0
        self.files={};self.closed=False
        self.durable_events=self.durable_bytes=self.commits=0
        self.digest=hashlib.sha256();self.final_receipt=None

    def commit(self,reason):
        if self.closed:raise ValueError('Open journal commit required')
        with operation('journal_fsync',reason):os.fsync(self.fd)
        # The journal directory is explicitly synced before acknowledging its
        # first durable prefix. No event after this prefix is claimed durable.
        if self.commits==0:
            fd=os.open(self.root,os.O_RDONLY)
            try:
                with operation('journal_directory_fsync',reason):os.fsync(fd)
            finally:os.close(fd)
        self.durable_events=self.sequence;self.durable_bytes=self.journal_bytes;self.commits+=1
        mark('journal_committed',reason,durable_events=self.durable_events,durable_bytes=self.durable_bytes)
        receipt=dict(schema=1,events=self.durable_events,bytes=self.durable_bytes,sha256=self.digest.hexdigest(),
                     commits=self.commits,maximum_pending_events=COMMIT_EVENTS,reason=reason,final=False)
        advisory_json(self.root/'journal-durable-prefix.json',receipt)
        return receipt

    def event(self, kind, **fields):
        if self.closed or self.sequence>=self.max_events:
            raise ValueError('Capture event budget/closed journal')
        if {'sequence','kind','seconds'} & fields.keys():
            raise ValueError('Reserved journal fields')
        row=dict(sequence=self.sequence,kind=kind,seconds=time.monotonic()-self.started,**fields)
        raw=(json.dumps(row,separators=(',',':'))+'\n').encode()
        if self.journal_bytes+len(raw)>self.max_journal_bytes:
            raise ValueError('Capture journal byte budget')
        view=memoryview(raw)
        with operation('journal_write',kind):
            while view:
                n=os.write(self.fd,view)
                if n<=0:raise OSError('Journal short write')
                view=view[n:]
        self.digest.update(raw)
        self.journal_bytes+=len(raw);self.sequence+=1
        mark('journal_written',kind,written_events=self.sequence,written_bytes=self.journal_bytes)
        if self.sequence-self.durable_events>=COMMIT_EVENTS or kind in COMMIT_FENCES:self.commit(kind)
        return row

    def save(self, label, raw, *, symbol, rectangle, dtype):
        if self.closed or not isinstance(raw,bytes):raise ValueError('Open byte capture required')
        if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in label):
            raise ValueError('Safe raw label')
        name=label+'.bin'
        if name in self.files or len(self.files)>=self.max_files:
            raise ValueError('Unique bounded raw files')
        if len(raw)>16<<20 or self.raw_bytes+len(raw)>self.max_raw_bytes:
            raise ValueError('Raw capture byte budget')
        x,y,width,height,count=rectangle
        if not (0<=x<119 and 0<=y<1160 and width>0 and height>0 and x+width<=119 and y+height<=1160 and count>0):
            raise ValueError('Application copy rectangle')
        element_bytes=2 if dtype=='u16' else 4
        if dtype not in ('u16','u32','f32') or len(raw)!=width*height*count*element_bytes:
            raise ValueError('Logical raw copy shape/dtype')
        path=self.raw/name
        with operation('raw_write',label):
            with path.open('xb') as f:
                f.write(raw);f.flush()
                with operation('raw_file_fsync',label):os.fsync(f.fileno())
        directory=os.open(self.raw,os.O_RDONLY)
        try:
            with operation('raw_directory_fsync',label):os.fsync(directory)
        finally:os.close(directory)
        pin=dict(path='raw/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            symbol=symbol,rectangle=list(rectangle),dtype=dtype)
        self.files[name]=pin;self.raw_bytes+=len(raw)
        self.event('raw_durable',receipt=pin)
        return pin

    def close(self):
        if not self.closed:
            receipt=self.commit('final_close')
            os.close(self.fd);self.closed=True
            receipt['final']=True
            atomic_json(self.root/'journal-durability.json',receipt)
            self.final_receipt=receipt
