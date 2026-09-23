"""Bounded post-release raw interface for an independent Layer3 numeric audit.

It exposes actual operands/results by serial/export/coordinate with no SDK or
reference-fed execution. Operator validators must separately establish exact
transport/bit identities and declared conditional arithmetic error bounds.
"""
from pathlib import Path
import hashlib
from runtime_plan import box

class Evidence:
    def __init__(self,root,capture,copies):
        self.root=Path(root);self.capture=capture;self.copies=copies;self.index={};self.read_bytes=0
        if capture['completed']!=[1,2,3] or capture['mathematical_audit_passed']:
            raise ValueError('Complete unaudited raw capture required')
        for serial in (1,2,3):
            for i,item in enumerate(copies['diagnostic_copies']):
                name=f's{serial}-diagnostic-{i:04}.bin';receipt=capture['raw_files'][name]
                if receipt['rectangle']!=box(item) or receipt['dtype']!=item['dtype'] or receipt['symbol']!=item['symbol']:
                    raise ValueError('Exact serial/export geometry provenance')
                for y in range(item['y'],item['y']+item['height']):
                    for x in range(item['x'],item['x']+item['width']):
                        key=(serial,item['symbol'],x,y)
                        if key in self.index:raise ValueError('Unique arithmetic raw ownership')
                        self.index[key]=(name,x-item['x'],y-item['y'])
        self.cached_name=None;self.cached_value=None

    def raw(self,name):
        import numpy as np
        r=self.capture['raw_files'][name];p=self.root/r['path']
        if Path(r['path']).parts!=('raw',name) or p.is_symlink() or p.stat().st_size!=r['bytes'] or r['bytes']>16<<20:
            raise ValueError('Bounded exact raw file')
        raw=p.read_bytes();self.read_bytes+=len(raw)
        if hashlib.sha256(raw).hexdigest()!=r['sha256']:raise ValueError('Captured raw bytes changed')
        x,y,w,h,n=r['rectangle'];dtype={'u16':'<u2','u32':'<u4','f32':'<f4'}[r['dtype']]
        a=np.frombuffer(raw,dtype=dtype)
        if a.size!=w*h*n:raise ValueError('Complete native raw shape')
        return a.reshape(h,w,n)

    def coordinate(self,serial,symbol,x,y):
        name,ix,iy=self.index[serial,symbol,x,y]
        if name!=self.cached_name:
            self.cached_value=self.raw(name);self.cached_name=name
        return self.cached_value[iy,ix].copy()

    def input(self,serial):return self.raw(f's{serial}-original-input.bin').reshape(5120).copy()

    def rectangle(self,serial,symbol,x,y,width,height,count):
        import numpy as np
        owners={}
        for yy in range(y,y+height):
            for xx in range(x,x+width):
                name,ix,iy=self.index[serial,symbol,xx,yy]
                owners.setdefault(name,[]).append((xx-x,yy-y,ix,iy))
        result=None
        for name,points in owners.items():
            value=self.raw(name)
            if value.shape[2]!=count:raise ValueError('Exact diagnostic rectangle element count')
            if result is None:result=np.empty((height,width,count),dtype=value.dtype)
            for xx,yy,ix,iy in points:result[yy,xx]=value[iy,ix]
        if result is None:raise ValueError('Nonempty actual raw rectangle')
        return result

