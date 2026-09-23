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
        while view:
            n=os.write(self.fd,view)
            if n<=0:raise OSError('Journal short write')
            view=view[n:]
        os.fsync(self.fd)
        self.journal_bytes+=len(raw);self.sequence+=1
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
        with path.open('xb') as f:
            f.write(raw);f.flush();os.fsync(f.fileno())
        directory=os.open(self.raw,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
        pin=dict(path='raw/'+name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
            symbol=symbol,rectangle=list(rectangle),dtype=dtype)
        self.files[name]=pin;self.raw_bytes+=len(raw)
        self.event('raw_durable',receipt=pin)
        return pin

    def close(self):
        if not self.closed:
            os.fsync(self.fd);os.close(self.fd);self.closed=True
