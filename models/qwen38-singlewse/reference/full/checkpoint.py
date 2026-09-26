"""Bounded, original FP8 tensor loading and explicit independent CPU W8A8 math.

This reference caches the 24.38 GB compressed language-layer weights in CPU RAM;
the large BF16 embedding/head tables are read by bounded row tiles. Candidate
device inference must keep all weights resident on-wafer. FP32 group partials
are summed in ascending K order.
"""
import hashlib,json,os
from pathlib import Path
import torch
from torch import nn

class BlockFP8Linear(nn.Module):
    def __init__(self,weight,scales):
        super().__init__();self.weight=weight;self.scales=scales.float()
        self.out_features,self.in_features=weight.shape
        assert self.in_features%128==0 and self.scales.shape==((self.out_features+127)//128,self.in_features//128)
    def forward(self,x):
        shape=x.shape;xf=x.float().reshape(-1,self.in_features//128,128)
        scales=xf.abs().amax(-1,keepdim=True).clamp_min(1e-10)*(1/448)
        quant=(xf/scales).clamp(-448,448).to(torch.float8_e4m3fn).float()
        result=torch.zeros((xf.shape[0],self.out_features),dtype=torch.float32)
        for k in range(self.in_features//128):
            weight=self.weight[:,k*128:(k+1)*128].float()
            partial=quant[:,k]@weight.T
            partial=(partial*scales[:,k])*self.scales[:,k].repeat_interleave(128)[:self.out_features]
            result+=partial
        return result.reshape(*shape[:-1],self.out_features).to(x.dtype)

class Checkpoint:
    def __init__(self,root,metadata):
        self.root=Path(root);self.meta=json.loads(Path(metadata).read_text())
        receipt=json.loads((self.root.parent/'AUDIT-COMPLETE.json').read_text())
        assert receipt['passed'] and receipt['revision']==self.meta['revision']
        self.pins={x['shard']:x['sha256'] for x in receipt['shards']}
        self.tensors={n:v for n,v in self.meta['tensors'].items() if n.startswith('model.language_model.') or n=='lm_head.weight'}
        assert len(self.tensors)==1251
        self.stamps={};self.verified=set();self.read_names=set();self.layer_cache={}
        resident=sum(v['bytes'] for n,v in self.tensors.items() if n.startswith('model.language_model.layers.'))
        assert resident<24<<30
        for v in self.tensors.values():
            p=self.root/v['shard'];assert not p.is_symlink();self.stamps[p]=self.stamp(p.stat())
    @staticmethod
    def stamp(s):return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    def check_unchanged(self):
        for p,s in self.stamps.items():
            if p.is_symlink() or self.stamp(p.stat())!=s:raise ValueError('Checkpoint changed')
    def tensor(self,name,first=None,count=None):
        v=self.tensors[name];p=self.root/v['shard'];self.check_unchanged()
        if name in self.layer_cache:
            assert first is None
            return self.layer_cache[name]
        if p not in self.verified:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(4<<20),b''):
                    h.update(chunk)
                    os.posix_fadvise(f.fileno(),f.tell()-len(chunk),len(chunk),os.POSIX_FADV_DONTNEED)
            assert h.hexdigest()==self.pins[p.name];self.verified.add(p)
        dtype={'BF16':torch.bfloat16,'F8_E4M3':torch.float8_e4m3fn}[v['dtype']]
        shape=v['shape'];start=v['data_start']+v['data_offsets'][0]
        if first is not None:
            assert len(shape)==2 and count is not None and 0<=first<first+count<=shape[0]
            start+=first*shape[1]*(2 if dtype==torch.bfloat16 else 1);shape=[count,shape[1]]
        mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        if int(mem['MemAvailable'].split()[0])<2*1024*1024:raise MemoryError('Reference preserves 2 GiB machine headroom')
        value=torch.empty(shape,dtype=dtype,device='cpu')
        view=memoryview(value.view(torch.uint8).numpy()).cast('B')
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            assert self.stamp(os.fstat(fd))==self.stamps[p]
            with os.fdopen(fd,'rb',buffering=0,closefd=False) as f:
                f.seek(start);offset=0
                while offset<len(view):
                    n=f.readinto(view[offset:offset+(4<<20)])
                    if not n:raise ValueError('Short weight read')
                    os.posix_fadvise(fd,start+offset,n,os.POSIX_FADV_DONTNEED)
                    offset+=n
        finally:os.close(fd)
        self.read_names.add(name)
        if name.startswith('model.language_model.layers.'):
            assert first is None;self.layer_cache[name]=value
        return value
    def bind_layer(self,module,index):
        prefix=f'model.language_model.layers.{index}.'
        expected={n[len(prefix):] for n in self.tensors if n.startswith(prefix) and not n.endswith('_scale_inv')}
        assert set(dict(module.named_parameters()))==expected
        quantized=[]
        for name,sub in list(module.named_modules()):
            key=prefix+name+'.weight'
            if isinstance(sub,nn.Linear) and self.tensors[key]['dtype']=='F8_E4M3':
                parent,_,leaf=name.rpartition('.')
                owner=module.get_submodule(parent) if parent else module
                setattr(owner,leaf,BlockFP8Linear(self.tensor(key),self.tensor(key+'_scale_inv')))
                quantized.append(name+'.')
        for name,param in list(module.named_parameters()):
            assert param.is_meta
            owner,_,leaf=name.rpartition('.')
            destination=module.get_submodule(owner) if owner else module
            tensor=self.tensor(prefix+name);assert tensor.shape==param.shape
            setattr(destination,leaf,nn.Parameter(tensor,requires_grad=False))
        return module.eval()
