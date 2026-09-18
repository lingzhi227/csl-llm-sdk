"""Read verified original BF16 tensors with bounded buffers and one layer in RAM."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat


def sha(path,check):
    h=hashlib.sha256()
    with path.open('rb',buffering=0) as f:
        while True:
            check();raw=f.read(1<<20)
            if not raw:break
            h.update(raw);os.posix_fadvise(f.fileno(),f.tell()-len(raw),len(raw),os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


class Checkpoint:
    def __init__(self,root,plan,headers,check):
        self.root=Path(root);self.plan=plan;self.headers=headers;self.check=check;self.stamps={};self.read_names=set()
        receipt=json.loads((self.root/'COMPLETE.json').read_bytes())
        if not(receipt['status']=='complete' and receipt['all_whole_shard_hashes_and_headers_verified']):
            raise ValueError('Completed original checkpoint required')
        for asset in plan['assets']:
            path=self.root/asset['file'];before=path.stat()
            if path.is_symlink() or before.st_size!=asset['bytes']:raise ValueError('Original asset size/type')
            if 'sha256' in asset:
                correct=sha(path,check)==asset['sha256']
            else:
                # Ordinary Git assets are individually bounded by pinned metadata.
                raw=path.read_bytes();check()
                correct=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==asset['git_blob_sha1']
            if not correct:raise ValueError('Original tokenizer/config content identity')
            self.stamps[path]=self.stamp(before)
        self.tensors={}
        for shard in plan['shards']:
            path=self.root/shard['file'];before=path.stat()
            if path.is_symlink() or before.st_size!=shard['bytes'] or not stat.S_ISREG(before.st_mode):raise ValueError('Shard type/size')
            if sha(path,check)!=shard['sha256']:raise ValueError('Original shard identity')
            self.stamps[path]=self.stamp(before)
            for name,meta in headers[shard['file']].items():
                if not(name.startswith('model.language_model.') or name=='lm_head.weight'):continue
                if name in self.tensors or meta['dtype']!='BF16':raise ValueError('Text tensor name/dtype')
                size=math.prod(meta['shape'])*2
                if size!=meta['data_offsets'][1]-meta['data_offsets'][0]:raise ValueError('Tensor extent')
                self.tensors[name]=dict(meta,path=path,start=shard['data_start']+meta['data_offsets'][0],bytes=size)
        if len(self.tensors)!=851 or sum(t['bytes'] for t in self.tensors.values())!=53791996928:
            raise ValueError('Full851 text tensor coverage')
        self.check_unchanged()

    @staticmethod
    def stamp(s):return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

    def check_unchanged(self):
        for p,s in self.stamps.items():
            if p.is_symlink() or self.stamp(p.stat())!=s:raise ValueError('Original shard changed')

    def tensor(self,name,first=None,count=None):
        import torch
        self.check();meta=self.tensors[name];shape=meta['shape'];start=meta['start']
        if first is not None:
            if len(shape)!=2 or count is None or not 0<=first<first+count<=shape[0]:raise ValueError('Bounded row extent')
            start+=first*shape[1]*2;shape=[count,shape[1]]
        value=torch.empty(shape,dtype=torch.bfloat16,device='cpu')
        view=memoryview(value.view(torch.int16).numpy()).cast('B');offset=0
        fd=os.open(meta['path'],os.O_RDONLY|os.O_NOFOLLOW)
        try:
            if self.stamp(os.fstat(fd))!=self.stamps[meta['path']]:raise ValueError('Shard changed before tensor read')
            with os.fdopen(fd,'rb',buffering=0,closefd=False) as stream:
                stream.seek(start)
                while offset<len(view):
                    self.check();n=stream.readinto(view[offset:offset+(1<<20)])
                    if not n:raise ValueError('Short original tensor')
                    os.posix_fadvise(fd,start+offset,n,os.POSIX_FADV_DONTNEED);offset+=n
        finally:os.close(fd)
        self.read_names.add(name);return value

    def bind_layer(self,module,index):
        import torch
        prefix=f'model.language_model.layers.{index}.'
        expected={n[len(prefix):] for n in self.tensors if n.startswith(prefix)}
        parameters=dict(module.named_parameters())
        if set(parameters)!=expected:raise ValueError('Original layer name coverage')
        for name,parameter in parameters.items():
            if not parameter.is_meta:raise ValueError('Avoid initialized original parameter copies')
            tensor=self.tensor(prefix+name)
            if tuple(tensor.shape)!=tuple(parameter.shape):raise ValueError('Original parameter shape')
            owner,_,leaf=name.rpartition('.');destination=module.get_submodule(owner) if owner else module
            setattr(destination,leaf,torch.nn.Parameter(tensor,requires_grad=False))
        if any(p.is_meta for p in module.parameters()):raise ValueError('Unbound meta parameter')
        return module.eval()
