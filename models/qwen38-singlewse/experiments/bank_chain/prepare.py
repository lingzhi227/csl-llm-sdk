"""Freeze a two-matrix device-chain oracle, including quantization intervals."""
import hashlib,json,struct
from pathlib import Path
import numpy as np

TINY=float(np.finfo(np.float32).tiny);GAMMA=300*2.0**-24/(1-300*2.0**-24)
def bits(x):
    b=np.asarray(x,np.float32).copy().view(np.uint32)
    return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
def expand(x):return (x.astype(np.uint32)<<np.uint32(16)).view(np.float32)
def bf(x):return expand(bits(x))
def decode(c):
    c=np.asarray(c,dtype=np.uint32);e=(c&127)>>3;m=c&7
    v=np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e.astype(np.int32)-7))
    return np.where(c&128,-v,v)
def encode(x):
    levels=decode(np.arange(127,dtype=np.uint8)).astype(np.float32)
    mid=(levels[:-1]+levels[1:])*np.float32(.5);magnitude=np.abs(x).clip(0,448)
    index=np.searchsorted(mid,magnitude,side='left');candidate=np.minimum(index,125)
    index+=(index<126)&(magnitude==mid[candidate])&(index%2==1)
    return index.astype(np.uint8)|(np.signbit(x).astype(np.uint8)<<7)
def quant(x):
    x=np.asarray(x,np.float32).reshape(-1,128)
    scale=np.maximum(np.max(np.abs(x),axis=1,keepdims=True),np.float32(1e-10))*np.float32(1/448)
    return decode(encode(x/scale))*scale.astype(np.float64)
def quant_interval(lo,hi):
    lo=np.asarray(lo,np.float32).reshape(-1,128);hi=np.asarray(hi,np.float32).reshape(-1,128)
    amin=np.where((lo<=0)&(hi>=0),0,np.minimum(np.abs(lo),np.abs(hi)))
    amax=np.maximum(np.abs(lo),np.abs(hi))
    sl=np.maximum(amin.max(axis=1,keepdims=True),np.float32(1e-10))*np.float32(1/448)
    sh=np.maximum(amax.max(axis=1,keepdims=True),np.float32(1e-10))*np.float32(1/448)
    ratios=np.stack([lo/sl,lo/sh,hi/sl,hi/sh])
    ql=decode(encode(ratios.min(axis=0)));qh=decode(encode(ratios.max(axis=0)))
    values=np.stack([ql*sl,qh*sl,ql*sh,qh*sh]);return values.min(axis=0),values.max(axis=0)
def silu(x):
    e=np.exp(-np.abs(x));return x*np.where(x<0,e/(1+e),1/(1+e))

def main():
    path=Path('/srv/qwen38-singlewse-hardware/fp8-matrix-hw-001/layers-0.safetensors')
    pin='07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
        assert h.hexdigest()==pin
        stream.seek(0);size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
        tiles=[];weights=[];scales=[];decoded=[]
        for row in range(4):
            matrix=1 if row<2 else 2;groups=2 if matrix==1 else 3;ordinal=row%2;first=[0,4,1,4][row]
            name='model.language_model.layers.0.mlp.'+('gate_proj' if matrix==1 else 'down_proj')+'.weight'
            w=header[name];s=header[name+'_scale_inv'];assert w['dtype']=='F8_E4M3' and s['dtype']=='BF16'
            stream.seek(8+size+s['data_offsets'][0]);all_scales=expand(np.frombuffer(stream.read(s['data_offsets'][1]-s['data_offsets'][0]),'<u2').copy()).reshape(s['shape'])
            for k in range(groups):
                raw=np.zeros((272,128),np.uint8)
                for r in range(272):
                    stream.seek(8+size+w['data_offsets'][0]+(ordinal*272+r)*w['shape'][1]+k*128)
                    raw[r]=np.frombuffer(stream.read(128),'u1')
                phase=ordinal*272%128;sc=all_scales[ordinal*272//128:ordinal*272//128+3,k].copy()
                weights.append(raw.T.copy().reshape(-1).view('<u2'));scales.append(sc)
                decoded.append(decode(raw)*sc[(phase+np.arange(272))//128,None].astype(np.float64))
                tiles.append(dict(x=first+k,y=row,matrix=matrix,k=k,original_row_tile=ordinal,tensor=name))
    i=np.arange(256,dtype=np.float64)
    inputs=bf(np.stack([np.sin(i*.17),((i%19)-9)/16,np.zeros(256)]))
    bank_lower=[];bank_upper=[];output_lower=[];output_upper=[];central=[]
    for x in inputs:
        a=np.zeros(544,np.float64);absolute=np.zeros_like(a);qa=quant(x)
        for tile,w in zip(tiles,decoded):
            if tile['matrix']!=1:continue
            at=slice(tile['original_row_tile']*272,(tile['original_row_tile']+1)*272);q=qa[tile['k']]
            a[at]+=w@q;absolute[at]+=np.abs(w)@np.abs(q)
        lo=bf(a-GAMMA*absolute-TINY);hi=bf(a+GAMMA*absolute+TINY);mid=bf(a).astype(np.float64)
        bank_lower.append(lo);bank_upper.append(hi)
        e=np.maximum(mid-lo,hi-mid);z=silu(mid)
        zl=bf(z-1.1*e-5e-6*np.abs(z)-2*TINY);zh=bf(z+1.1*e+5e-6*np.abs(z)+2*TINY)
        ql,qh=quant_interval(zl[:384],zh[:384]);qm=quant(bf(z[:384]))
        bl=np.zeros(544,np.float64);bh=np.zeros_like(bl);bm=np.zeros_like(bl)
        for tile,w in zip(tiles,decoded):
            if tile['matrix']!=2:continue
            k=tile['k'];at=slice(tile['original_row_tile']*272,(tile['original_row_tile']+1)*272)
            pos=np.maximum(w,0);neg=np.minimum(w,0)
            lower=pos@ql[k]+neg@qh[k];upper=pos@qh[k]+neg@ql[k]
            rounding=GAMMA*(np.abs(w)@np.maximum(np.abs(ql[k]),np.abs(qh[k])))+TINY
            bl[at]+=lower-rounding;bh[at]+=upper+rounding;bm[at]+=w@qm[k]
        output_lower.append(bl);output_upper.append(bh);central.append(bm)
    assert np.max(np.array(output_upper)-np.array(output_lower))<.01
    np.savez('fixture.npz',weights=np.array(weights),scales=np.array(scales),inputs=inputs,
        bank_lower=np.array(bank_lower),bank_upper=np.array(bank_upper),output_lower=np.array(output_lower),
        output_upper=np.array(output_upper),central=np.array(central))
    meta=dict(scope='Original gate544x256 -> resident BF16 bank -> SiLU/BF16/dynamicFP8 -> original down544x384',
        full_model=False,full_mlp=False,shard_sha256=pin,tiles=tiles,application_pes=45,command_frames_per_call=21,
        bank_capacity_elements=17536,criterion='Frozen propagated FP64/GEMV/BF16/SiLU/FP8-scale-and-code intervals; device routes/counters; original retention; zero bank tail',
        maximum_output_interval_width=float(np.max(np.array(output_upper)-np.array(output_lower))),
        fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest())
    Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
