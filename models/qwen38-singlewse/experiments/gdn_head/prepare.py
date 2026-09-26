"""Freeze an analytic FP64 pipeline oracle with propagated input/state intervals.

Original layer0 head0 parameters; synthetic BF16 projected operands. Bounds
include qualified scalar nonlinear approximation, all BF16 discontinuities,
normalization reductions and accumulated state uncertainty. No CSL is imported.
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np

U=2.0**-24;TINY=float(np.finfo(np.float32).tiny)
def gamma(n):return n*U/(1-n*U)
def bits(x):
    raw=np.asarray(x,np.float32).copy().view(np.uint32)
    return ((raw+np.uint32(0x7fff)+((raw>>16)&1))>>16).astype(np.uint16)
def expand(v):return np.left_shift(np.asarray(v,dtype=np.uint16).astype(np.uint32),np.uint32(16)).view(np.float32)
def bf(x):return expand(bits(x)).astype(np.float64)
def sigmoid(x):
    e=np.exp(-np.abs(x));return np.where(x<0,e/(1+e),1/(1+e))
def silu(x):return x*sigmoid(x)
def rounded_error(point,error):
    mid=bf(point);lo=bf(point-error);hi=bf(point+error)
    return mid,np.maximum(mid-lo,hi-mid)
def product_interval(lo,hi,a,b):
    v=np.stack([lo*a,lo*b,hi*a,hi*b]);return v.min(axis=0),v.max(axis=0)
def normalized_interval(x,error,mean):
    lo=x-error;hi=x+error
    square_min=np.where((lo<=0)&(hi>=0),0,np.minimum(lo*lo,hi*hi))
    square_max=np.maximum(lo*lo,hi*hi)
    reduction=np.mean if mean else np.sum
    inverse_lo=1/np.sqrt(reduction(square_max)+1e-6)
    inverse_hi=1/np.sqrt(reduction(square_min)+1e-6)
    low,high=product_interval(lo,hi,inverse_lo,inverse_hi)
    extra=gamma(260)*np.maximum(np.abs(low),np.abs(high))+TINY
    return low-extra,high+extra

def main():
    source=Path('/srv/qwen38-singlewse-hardware/fp8-matrix-hw-001/layers-0.safetensors')
    pin='07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'
    h=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
        assert h.hexdigest()==pin
        stream.seek(0);size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
        def read(suffix):
            spec=header['model.language_model.layers.0.linear_attn.'+suffix]
            assert spec['dtype']=='BF16';first,last=spec['data_offsets']
            stream.seek(8+size+first)
            return np.frombuffer(stream.read(last-first),dtype='<u2').copy().reshape(spec['shape'])
        conv=read('conv1d.weight').reshape(10240,4)
        conv=conv[np.r_[np.arange(128),2048+np.arange(128),4096+np.arange(128)]]
        params=np.array([read('A_log')[0],read('dt_bias')[0]],np.uint16)
        gain=read('norm.weight')
    rng=np.random.default_rng(380028)
    inputs=bits(rng.normal(size=(96,514)))
    # Choose meaningful decays using the original learned parameters. Generic
    # small a values can produce near-unit decay and useless absolute interval
    # growth; reject such an oracle before any candidate device observations.
    target_decay_log=rng.uniform(.6,1.4,96)
    A_log,dt_bias=expand(params).astype(np.float64)
    inputs[:,512]=bits(np.log(np.expm1(target_decay_log/np.exp(A_log)))-dt_bias)
    inputs[0,:512]=0;inputs[31,513]=bits(-20.0);inputs[63,513]=bits(20.0)
    weights=expand(conv).astype(np.float64);A_log,dt_bias=expand(params).astype(np.float64)
    norm_gain=expand(gain).astype(np.float64)
    history=np.zeros((384,4),np.float64);history_bits=np.zeros((384,4),np.uint16)
    state=np.zeros((128,128),np.float64);state_error=np.zeros_like(state)
    records={k:[] for k in ['packet','packet_bound','state','state_bound','output','output_bound','gated_lower','gated_upper','history']}
    for raw in inputs:
        x=expand(raw).astype(np.float64)
        history[:,:3]=history[:,1:];history[:,3]=x[:384]
        history_bits[:,:3]=history_bits[:,1:];history_bits[:,3]=raw[:384]
        conv_exact=np.sum(history*weights,axis=1)
        conv_bound=gamma(8)*np.sum(np.abs(history*weights),axis=1)+TINY
        convolved,conv_error=rounded_error(conv_exact,conv_bound)
        activated=silu(convolved)
        # |SiLU'(x)| <1.1 on the real line, including its nonmonotone tail.
        activated,activation_error=rounded_error(activated,1.1*conv_error+5e-6*np.abs(activated)+2*TINY)
        packet=np.zeros(386,np.float64);pe=np.zeros(386,np.float64)
        for head in range(2):
            at=slice(head*128,(head+1)*128);v=activated[at]
            packet[at]=v/np.sqrt(np.sum(v*v)+1e-6)
            lo,hi=normalized_interval(v,activation_error[at],False)
            pe[at]=np.maximum(packet[at]-lo,hi-packet[at])
        packet[:128]/=np.sqrt(128);pe[:128]=pe[:128]/np.sqrt(128)+U*np.abs(packet[:128])+TINY
        packet[256:384]=activated[256:384];pe[256:384]=activation_error[256:384]
        g=-np.exp(A_log)*np.logaddexp(0,x[512]+dt_bias)
        ge=6e-6*abs(g)+2*TINY
        packet[384]=np.exp(g)
        pe[384]=max(np.exp(g+ge)*(1+2e-6)-packet[384],packet[384]-np.exp(g-ge)*(1-2e-6))+2*TINY
        packet[385],pe[385]=rounded_error(sigmoid(x[513]),2e-6*sigmoid(x[513])+2*TINY)
        q,k,v=packet[:128],packet[128:256],packet[256:384]
        qe,ke,ve=pe[:128],pe[128:256],pe[256:384]
        decay,beta=packet[384:];de,be=pe[384:]
        decayed=state*decay
        db=(abs(decay)+de)*state_error+de*np.abs(state)
        db=(1+U)*db+U*np.abs(decayed)+TINY
        prediction=k@decayed
        pb=np.abs(k)@db+ke@(np.abs(decayed)+db)
        pb+=gamma(256)*((np.abs(k)+ke)@(np.abs(decayed)+db))+256*TINY
        difference=v-prediction;difference_error=(1+U)*(ve+pb)+U*np.abs(difference)+TINY
        delta=difference*beta
        delta_error=(1+U)*((abs(beta)+be)*difference_error+be*np.abs(difference))+U*np.abs(delta)+TINY
        correction=k[:,None]*delta
        cb=(1+U)*((np.abs(k[:,None])+ke[:,None])*delta_error+ke[:,None]*np.abs(delta))+U*np.abs(correction)+TINY
        state=decayed+correction;state_error=(1+U)*(db+cb)+U*np.abs(state)+TINY
        output=q@state
        oe=np.abs(q)@state_error+qe@(np.abs(state)+state_error)
        oe+=gamma(256)*((np.abs(q)+qe)@(np.abs(state)+state_error))+256*TINY
        core,core_error=rounded_error(output,oe)
        lo,hi=normalized_interval(core,core_error,True)
        lo,hi=bf(lo),bf(hi)
        lo,hi=product_interval(lo,hi,norm_gain,norm_gain);lo,hi=bf(lo),bf(hi)
        z=silu(x[384:512]);ze=5e-6*np.abs(z)+2*TINY
        lo,hi=product_interval(lo,hi,z-ze,z+ze)
        extra=gamma(2)*np.maximum(np.abs(lo),np.abs(hi))+TINY
        data=dict(packet=packet,packet_bound=pe,state=state,state_bound=state_error,output=output,
                  output_bound=oe,gated_lower=bf(lo-extra),gated_upper=bf(hi+extra),history=history_bits)
        for key,value in data.items():records[key].append(value.copy())
    records={k:np.array(v) for k,v in records.items()}
    assert all(np.isfinite(v).all() for v in records.values())
    assert records['state_bound'].max()<.1 and records['output_bound'].max()<.02
    assert (records['gated_upper']-records['gated_lower']).max()<.5
    np.savez_compressed('fixture.npz',inputs=inputs,conv_weights=conv.reshape(-1),gate_parameters=params,gain=gain,**records)
    meta=dict(scope='Original layer0 head0 convolution/gates/norm plus full128x128 recurrent state; synthetic BF16 projected operands',
        shard_sha256=pin,seed=380028,positions=96,reset_replay_positions=8,full_model=False,
        reference='Pinned Transformers arithmetic; independent NumPy FP64 analytic oracle, propagated BF16/scalar/reduction/state intervals',
        maximum_packet_bound=float(records['packet_bound'].max()),maximum_state_bound=float(records['state_bound'].max()),
        maximum_output_bound=float(records['output_bound'].max()),maximum_gated_interval_width=float((records['gated_upper']-records['gated_lower']).max()),
        fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest())
    Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
