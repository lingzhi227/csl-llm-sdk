"""Bounded original-tensor packing for resident initialization, never token steps.

One output-row strip is read at a time. The caller must bind these arrays to a
compiled, role-checked upload plan. This module does not launch or upload to SDK.
"""
import hashlib,json,math,os,struct
from pathlib import Path
import numpy as np

class OriginalWeights:
    def __init__(self,root,tensor_metadata,shard_sha256):
        self.root=Path(root);self.tensors=tensor_metadata;self.pins=shard_sha256
        self.verified={}
    @staticmethod
    def stamp(s):return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    def verify_shard(self,name):
        p=self.root/name
        if p.is_symlink():raise ValueError('Checkpoint symlink')
        stamp=self.stamp(p.stat())
        if name in self.verified:
            if self.verified[name]!=stamp:raise ValueError('Checkpoint changed')
            return p
        h=hashlib.sha256()
        with p.open('rb') as f:
            for raw in iter(lambda:f.read(4<<20),b''):h.update(raw)
            f.seek(0);size=struct.unpack('<Q',f.read(8))[0]
            if size>4<<20:raise ValueError('Tensor header bound')
            header=json.loads(f.read(size))
        if h.hexdigest()!=self.pins[name] or self.stamp(p.stat())!=stamp:raise ValueError('Publisher hash failed')
        for tensor,meta in self.tensors.items():
            if meta['shard']==name:
                if header[tensor]!={k:meta[k] for k in ['dtype','shape','data_offsets']} or meta['data_start']!=size+8:
                    raise ValueError('Tensor header differs from pinned metadata')
        self.verified[name]=stamp;return p
    def rows(self,name,first,count):
        meta=self.tensors[name];shape=meta['shape'];dtype=meta['dtype']
        if len(shape)!=2 or not 0<=first<first+count<=shape[0]:raise ValueError('Row extent')
        if dtype not in ['BF16','F8_E4M3']:raise ValueError('Unsupported weight dtype')
        size=2 if dtype=='BF16' else 1;length=count*shape[1]*size
        if length>8<<20:raise ValueError('Resident initialization read bound')
        p=self.verify_shard(meta['shard']);start=meta['data_start']+meta['data_offsets'][0]+first*shape[1]*size
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            if self.stamp(os.fstat(fd))!=self.verified[meta['shard']]:raise ValueError('Checkpoint changed during open')
            raw=os.pread(fd,length,start)
        finally:os.close(fd)
        if len(raw)!=length:raise ValueError('Short tensor read')
        return np.frombuffer(raw,dtype='<u2' if size==2 else 'u1').reshape(count,shape[1])
    def small_tensor(self,name):
        """Original other-weight bytes, including convolution's3-D shape."""
        meta=self.tensors[name]
        if meta['dtype']!='BF16' or not 0<meta['bytes']<=131072:
            raise ValueError('Small original BF16 tensor profile')
        path=self.verify_shard(meta['shard']);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            if self.stamp(os.fstat(fd))!=self.verified[meta['shard']]:raise ValueError('Checkpoint changed during small-tensor open')
            raw=os.pread(fd,meta['bytes'],meta['data_start']+meta['data_offsets'][0])
        finally:os.close(fd)
        if len(raw)!=meta['bytes']:raise ValueError('Short small-tensor read')
        return np.frombuffer(raw,dtype='<u2').reshape(meta['shape'])
    def strip(self,name,tile_row,*,bf16_rows=144):
        meta=self.tensors[name];m,k=meta['shape'];fp8=meta['dtype']=='F8_E4M3'
        if bf16_rows not in [140,144]:raise ValueError('Unqualified BF16 tile geometry')
        rows=272 if fp8 else bf16_rows
        if k%128 or not 0<=tile_row<math.ceil(m/rows):raise ValueError('Matrix tile extent')
        first=tile_row*rows;valid=min(rows,m-first)
        values=self.rows(name,first,valid)
        padded=np.zeros((rows,k),dtype=values.dtype);padded[:valid]=values
        # Explicit contraction tiles are independent of the full-matrix fixture's
        # four-dimensional reshape/transpose packing implementation.
        packed=[]
        for column in range(k//128):
            block=np.ascontiguousarray(padded[:,column*128:(column+1)*128].T)
            packed.append(block.reshape(-1).view('<u2').copy() if fp8 else block.reshape(-1).copy())
        scales=None
        if fp8:
            if np.any((values&127)==127):raise ValueError('FP8 NaN weight')
            scale_name=name+'_scale_inv';scale_meta=self.tensors[scale_name]
            start=first//128;count=min(3,scale_meta['shape'][0]-start)
            original=self.rows(scale_name,start,count)
            scales=np.ones((k//128,3),np.float32)
            scales[:,:count]=(original.astype(np.uint32)<<16).view(np.float32).T
            if not np.isfinite(scales).all() or not np.all(scales>0):raise ValueError('Invalid original block scale')
        result=dict(name=name,tile_row=tile_row,valid_rows=valid,row_phase=first%128 if fp8 else None,
                    weights=np.stack(packed),scales=scales)
        if sum(v.nbytes for v in result.values() if isinstance(v,np.ndarray))>8<<20:raise ValueError('Packed strip bound')
        return result
