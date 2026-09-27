"""Freeze a FP64 full-head oracle and propagated rounding bounds before hardware.

The formula follows the pinned Transformers recurrent_gated_delta_rule after
Q/K normalization and exp(g). No candidate CSL implementation is imported.
Inputs and beta are representable at the caller's declared FP32/BF16 boundary.
Bounds propagate previous state uncertainty through every dependent position;
they include unfused products/adds, so also cover the CSL FMA accumulation.
"""
import hashlib,json
from pathlib import Path
import numpy as np

def bf16(x):
    x=np.asarray(x,np.float32).copy();bits=x.view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)

def main():
    rng=np.random.default_rng(380027)
    packet=np.zeros((96,386),np.float32)
    for t in range(96):
        q,k=bf16(rng.normal(size=(2,128))).astype(np.float64)
        packet[t,:128]=(q/np.sqrt(np.sum(q*q)+1e-6)/np.sqrt(128)).astype(np.float32)
        packet[t,128:256]=(k/np.sqrt(np.sum(k*k)+1e-6)).astype(np.float32)
        packet[t,256:384]=bf16(rng.normal(size=128))
        packet[t,384]=np.float32(np.exp(-rng.uniform(.01,1.2)))
        packet[t,385]=bf16([rng.uniform(.01,.99)])[0]
    packet[0,:384]=0
    packet[8,384:]=[1,0]  # Exact state retention without a correction.
    packet[16,384]=0      # History must be discarded by the recurrence.
    packet[31,385]=1
    packet[47,:256]=0
    unit=2.0**-24;gamma=lambda n:n*unit/(1-n*unit)
    tiny=float(np.finfo(np.float32).tiny)
    state=np.zeros((128,128),np.float64);bound=np.zeros_like(state)
    states=[];state_bounds=[];outputs=[];output_bounds=[]
    for raw in packet:
        p=raw.astype(np.float64);q,k,v=p[:128],p[128:256],p[256:384];decay,beta=p[384:]
        decayed=state*decay
        decayed_bound=(1+unit)*abs(decay)*bound+unit*np.abs(decayed)+tiny
        prediction=np.einsum('kv,k->v',decayed,k)
        prediction_bound=np.einsum('kv,k->v',decayed_bound,np.abs(k))
        prediction_bound+=gamma(256)*np.einsum('kv,k->v',np.abs(decayed)+decayed_bound,np.abs(k))+256*tiny
        difference=v-prediction
        difference_bound=(1+unit)*prediction_bound+unit*np.abs(difference)+tiny
        delta=difference*beta
        delta_bound=(1+unit)*abs(beta)*difference_bound+unit*np.abs(delta)+tiny
        correction=k[:,None]*delta
        correction_bound=(1+unit)*np.abs(k[:,None])*delta_bound+unit*np.abs(correction)+tiny
        state=decayed+correction
        bound=(1+unit)*(decayed_bound+correction_bound)+unit*np.abs(state)+tiny
        output=np.einsum('kv,k->v',state,q)
        ob=np.einsum('kv,k->v',bound,np.abs(q))
        ob+=gamma(256)*np.einsum('kv,k->v',np.abs(state)+bound,np.abs(q))+256*tiny
        states.append(state.copy());state_bounds.append(bound.copy())
        outputs.append(output);output_bounds.append(ob)
    np.savez_compressed('fixture.npz',packet=packet,state=np.array(states),state_bound=np.array(state_bounds),
        output=np.array(outputs),output_bound=np.array(output_bounds))
    meta=dict(scope='Full128x128 FP32 recurrent state; preprocessed synthetic operands',
        source='Transformers27166ea03f12c940f23176a904ab1d2ff1a3dcbb torch_recurrent_gated_delta_rule',
        positions=96,reset_replay_positions=8,seed=380027,physical=False,
        tolerance='FP64 oracle with a priori propagated IEEE FP32 gamma(256) reduction and per-operation bounds; includes prior-state error',
        maximum_state_bound=float(np.max(state_bounds)),maximum_output_bound=float(np.max(output_bounds)),
        fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest())
    Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))

if __name__=='__main__':main()
