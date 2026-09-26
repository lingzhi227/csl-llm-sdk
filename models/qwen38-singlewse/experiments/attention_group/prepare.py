"""Original Q/K norm gains and analytic interval oracle for a complete KV group.

The oracle never imports CSL or reads device outputs. All256 channels and all
past positions participate in each of six query heads. FP32/BF16 boundaries
follow pinned eager Transformers, including partial64 rotary and sigmoid gate.
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np
import torch

U=2.0**-24;TINY=float(np.finfo(np.float32).tiny)
def gamma(n):return n*U/(1-n*U)
def bits(x):
    b=np.asarray(x,np.float32).copy().view(np.uint32)
    return ((b+np.uint32(0x7fff)+((b>>16)&1))>>16).astype(np.uint16)
def expand(x):return np.left_shift(np.asarray(x,np.uint16).astype(np.uint32),np.uint32(16)).view(np.float32)
def bf(x):return expand(bits(x)).astype(np.float64)
def product(lo,hi,a,b):
    products=np.stack([lo*a,lo*b,hi*a,hi*b]);return products.min(axis=0),products.max(axis=0)
def sigmoid(x):
    e=np.exp(-np.abs(x));return np.where(x<0,e/(1+e),1/(1+e))

def norm_rotary(x,gain,frequencies,position):
    exact=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*(1+gain)
    error=gamma(530)*np.abs(exact)+TINY
    lo,hi=bf(exact-error),bf(exact+error)
    angles=(frequencies*np.float32(position)).astype(np.float32).astype(np.float64)
    # Explicit short-context SDK trigonometric acceptance profile. The result
    # after every BF16 multiply and add is independently interval-checked.
    cl,ch=bf(np.cos(angles)-1e-6),bf(np.cos(angles)+1e-6)
    sl,sh=bf(np.sin(angles)-1e-6),bf(np.sin(angles)+1e-6)
    left_lo,left_hi=lo[...,:32].copy(),hi[...,:32].copy()
    right_lo,right_hi=lo[...,32:64].copy(),hi[...,32:64].copy()
    a,b=product(left_lo,left_hi,cl,ch);c,d=product(-right_hi,-right_lo,sl,sh)
    lo[...,:32]=bf(bf(a)+bf(c));hi[...,:32]=bf(bf(b)+bf(d))
    a,b=product(right_lo,right_hi,cl,ch);c,d=product(left_lo,left_hi,sl,sh)
    lo[...,32:64]=bf(bf(a)+bf(c));hi[...,32:64]=bf(bf(b)+bf(d))
    return lo,hi

def main():
    torch.set_num_threads(1)
    source=Path('/srv/model-storage/qwen38-singlewse/model/layers-3.safetensors')
    pin='302f9af90bb683a8be9e96d124b470a2eddee6612c95c39a8e93f26eb654563d'
    h=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
        assert h.hexdigest()==pin
        stream.seek(0);size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
        def read(name):
            spec=header['model.language_model.layers.3.self_attn.'+name+'.weight']
            assert spec['shape']==[256] and spec['dtype']=='BF16'
            stream.seek(8+size+spec['data_offsets'][0]);return np.frombuffer(stream.read(512),'<u2').copy()
        gains=np.r_[read('q_norm'),read('k_norm')]
    # Exactly the pinned default-RoPE initialization, including FP32 pow/divide.
    frequencies=(1.0/(10000000**(torch.arange(0,64,2,dtype=torch.float32)/64))).numpy().copy()
    rng=np.random.default_rng(380029)
    inputs=bits(rng.normal(size=(96,3584)))
    inputs[0,:256]=0;inputs[0,512:2048]=0
    gain=expand(gains).astype(np.float64)
    values=[];key_lows=[];key_highs=[];pre_lows=[];pre_highs=[];out_lows=[];out_highs=[]
    for position,raw in enumerate(inputs):
        x=expand(raw).astype(np.float64)
        kl,kh=norm_rotary(x[:256],gain[256:],frequencies,position)
        ql,qh=norm_rotary(x[512:2048].reshape(6,256),gain[:256],frequencies,position)
        key_lows.append(kl);key_highs.append(kh);values.append(x[256:512])
        pre_lows.append(np.r_[kl,ql.reshape(-1)]);pre_highs.append(np.r_[kh,qh.reshape(-1)])
        # Cross-shard FP32 reduction followed by the original BF16 matmul cast.
        pl,ph=product(ql[:,None,:],qh[:,None,:],np.array(key_lows)[None,:,:],np.array(key_highs)[None,:,:])
        rounding=gamma(520)*np.maximum(np.abs(pl),np.abs(ph)).sum(axis=-1)+520*TINY
        ll=bf(bf(pl.sum(axis=-1)-rounding)*.0625)
        lh=bf(bf(ph.sum(axis=-1)+rounding)*.0625)
        # Tight monotonic softmax bounds use each component's own endpoint and
        # the opposite endpoints of every other causal logit.
        maximum=lh.max(axis=-1,keepdims=True)
        el=np.exp(ll-maximum);eh=np.exp(lh-maximum)
        low=el/(el+eh.sum(axis=-1,keepdims=True)-eh)
        high=eh/(eh+el.sum(axis=-1,keepdims=True)-el)
        low=bf(np.maximum(0,low*(1-3e-5)-2*TINY))
        high=bf(np.minimum(1,high*(1+3e-5)+2*TINY))
        v=np.array(values)[None,:,:]
        pl,ph=product(low[:,:,None],high[:,:,None],v,v)
        rounding=gamma(200)*np.maximum(np.abs(pl),np.abs(ph)).sum(axis=1)+200*TINY
        cl=bf(pl.sum(axis=1)-rounding);ch=bf(ph.sum(axis=1)+rounding)
        gate=sigmoid(x[2048:].reshape(6,256))
        gl=bf(gate*(1-2e-6)-2*TINY);gh=bf(gate*(1+2e-6)+2*TINY)
        lo,hi=product(cl,ch,gl,gh)
        error=gamma(2)*np.maximum(np.abs(lo),np.abs(hi))+TINY
        out_lows.append(bf(lo-error));out_highs.append(bf(hi+error))
    out_lows=np.array(out_lows);out_highs=np.array(out_highs)
    assert np.isfinite(out_lows).all() and np.isfinite(out_highs).all()
    assert np.max(out_highs-out_lows)<.25
    np.savez_compressed('fixture.npz',inputs=inputs,gain=gains,frequencies=frequencies,
        preprocess_lower=np.array(pre_lows),preprocess_upper=np.array(pre_highs),
        key_lower=np.array(key_lows),key_upper=np.array(key_highs),value_bits=inputs[:,256:512],
        output_lower=out_lows,output_upper=out_highs)
    meta=dict(scope='Original layer3 QK norm; one complete KV group with6 query heads, partial64 RoPE,96-position cache,causal softmax,sigmoid gate; synthetic projected BF16 operands',
        shard_sha256=pin,seed=380029,positions=96,reset_replay_positions=4,full_model=False,
        reference='Independent FP64 propagated intervals around eager BF16 boundaries; SDK trig absolute1e-6 and previously qualified nonlinear profiles',
        maximum_output_interval_width=float((out_highs-out_lows).max()),torch_version=torch.__version__,
        frequency_uint32=frequencies.view(np.uint32).tolist(),
        fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest())
    Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
