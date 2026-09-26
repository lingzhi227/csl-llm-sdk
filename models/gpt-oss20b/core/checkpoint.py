"""Bounded, read-only slices of the original Hugging Face safetensors shards."""
import json
from pathlib import Path
import struct
import numpy as np


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}

    def tensor(self, name):
        shard = self.index[name]
        path = self.root / shard
        if shard not in self.headers:
            with path.open("rb") as stream:
                n, = struct.unpack("<Q", stream.read(8))
                if not 0 < n < 8 << 20:
                    raise ValueError("Invalid safetensors header size")
                header = json.loads(stream.read(n))
            self.headers[shard] = (8+n, header)
        base, header = self.headers[shard]
        spec = header[name]
        dtype = {"U8": np.dtype("u1"), "BF16": np.dtype("<u2")}[spec["dtype"]]
        a, b = spec["data_offsets"]
        if b-a != int(np.prod(spec["shape"])) * dtype.itemsize or base+b > path.stat().st_size:
            raise ValueError("Tensor extent mismatch")
        return np.memmap(path, mode="r", dtype=dtype, offset=base+a, shape=tuple(spec["shape"]))

    def row_slice(self,name,start,count=1):
        """Read a bounded first-axis slice without mapping a whole vocabulary matrix."""
        path=self.root/self.index[name]
        if self.index[name] not in self.headers:
            with path.open('rb') as stream:
                n,=struct.unpack('<Q',stream.read(8))
                if not 0<n<8<<20:raise ValueError('Invalid safetensors header size')
                self.headers[self.index[name]]=(8+n,json.loads(stream.read(n)))
        base,header=self.headers[self.index[name]];spec=header[name]
        dtype={'U8':np.dtype('u1'),'BF16':np.dtype('<u2')}[spec['dtype']]
        shape=tuple(spec['shape']);a,b=spec['data_offsets']
        if not shape or not 0<=start<start+count<=shape[0]:raise ValueError('Invalid row slice')
        row_bytes=int(np.prod(shape[1:],dtype=np.int64))*dtype.itemsize
        if b-a!=shape[0]*row_bytes or base+b>path.stat().st_size:raise ValueError('Tensor extent mismatch')
        if row_bytes*count>1<<20:raise ValueError('Row slice exceeds bounded 1 MiB mapping')
        return np.memmap(path,mode='r',dtype=dtype,offset=base+a+start*row_bytes,shape=(count,*shape[1:]))


def bf16_float(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def float_bf16(values):
    values = np.asarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    if not np.isfinite(values).all():
        raise ValueError("Finite model boundary required")
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


FP4 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=np.float32)


def decode_mxfp4(blocks, scales):
    blocks, scales = np.asarray(blocks), np.asarray(scales)
    if blocks.dtype != np.uint8 or scales.dtype != np.uint8 or blocks.shape[:-1] != scales.shape or blocks.shape[-1] != 16:
        raise ValueError("MXFP4 requires 16 packed bytes and one scale per block")
    if np.any(scales == 255):
        raise ValueError("Reserved E8M0 NaN scale")
    out = np.empty((*scales.shape, 32), dtype=np.float32)
    out[..., 0::2] = FP4[blocks & 15]
    out[..., 1::2] = FP4[blocks >> 4]
    return np.ldexp(out, scales.astype(np.int32)[..., None] - 127)
