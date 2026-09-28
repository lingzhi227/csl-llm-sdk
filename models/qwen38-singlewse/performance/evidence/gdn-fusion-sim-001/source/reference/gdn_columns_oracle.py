"""Independent FP64 recurrence with a priori FP32 forward-error envelopes.

Inputs are exact supplied FP32 packets. The reference uses V minus a completed
dot product, not the candidate's reassociated update. Errors propagate across
all prior states. Neither CSL nor candidate outputs influence these intervals.
"""
import numpy as np
from reference.frontend_oracle import U,TINY,gamma,bits,expand


def step(state,error,packet):
    p=np.asarray(packet,dtype=np.float32).astype(np.float64)
    if p.shape!=(386,) or state.shape!=(128,128) or error.shape!=state.shape:
        raise ValueError('Complete original head and exact FP32 packet required')
    decay,beta=p[:2];v=p[2:130];k=p[130:258];q=p[258:]
    if not np.isfinite(p).all() or not 0<=decay<=1 or not 0<=beta<=1:
        raise ValueError('Finite recurrent gates required')
    decayed=state*decay
    de=error*abs(decay)+U*(np.abs(state)+error)*abs(decay)+TINY
    residual=v-k@decayed
    re=np.abs(k)@de+gamma(128)*(np.abs(v)+np.abs(k)@(np.abs(decayed)+de))+128*TINY
    delta=residual*beta
    delta_error=re*abs(beta)+U*(np.abs(residual)+re)*abs(beta)+TINY
    result=decayed+k[:,None]*delta
    bound=de+np.abs(k[:,None])*delta_error
    bound+=U*(np.abs(decayed)+de+np.abs(k[:,None])*(np.abs(delta)+delta_error))+TINY
    output=q@result
    output_error=np.abs(q)@bound+gamma(128)*(np.abs(q)@(np.abs(result)+bound))+128*TINY
    low=expand(bits(output-output_error)).astype(np.float64)
    high=expand(bits(output+output_error)).astype(np.float64)
    if not all(np.isfinite(a).all() for a in (result,bound,output,output_error,low,high)):
        raise ValueError('Nonfinite frozen recurrent reference')
    ratio=float(np.linalg.norm(bound)/max(np.linalg.norm(result),1e-30))
    if ratio>0.002:raise ValueError('A priori state error envelope exceeds qualification cap')
    return result,bound,low,high,ratio
