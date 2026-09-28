"""Independent FP64 frontend intervals; no candidate CSL is imported.

Scalar/BF16 interval helpers originate in the preserved original-parameter
GDN-head reference. This module covers shared Q/K state and all three V heads.
"""
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


def frontend_step(raw, weights, params, gain, history):
    """640 Q/K/V channels,384 Z values,3 A and3 B, all original BF16."""
    x=expand(raw).astype(np.float64);w=expand(weights).astype(np.float64)
    full=np.concatenate((history,raw[:640,None]),axis=1)
    history[:,:2]=history[:,1:];history[:,2]=raw[:640]
    h=expand(full).astype(np.float64)
    convolution=np.sum(h*w,axis=1)
    cb=gamma(8)*np.sum(np.abs(h*w),axis=1)+TINY
    conv,ce=rounded_error(convolution,cb)
    activation=silu(conv)
    activation,ae=rounded_error(activation,1.1*ce+5e-6*np.abs(activation)+2*TINY)
    shared=np.zeros(256,np.float64);se=np.zeros(256,np.float64)
    for head in range(2):
        at=slice(128*head,128*(head+1));v=activation[at]
        shared[at]=v/np.sqrt(np.sum(v*v)+1e-6)
        lo,hi=normalized_interval(v,ae[at],False)
        se[at]=np.maximum(shared[at]-lo,hi-shared[at])
    shared[:128]/=np.sqrt(128);se[:128]=se[:128]/np.sqrt(128)+U*np.abs(shared[:128])+TINY
    a_log=expand(params[:3]).astype(np.float64);bias=expand(params[3:]).astype(np.float64)
    g=-np.exp(a_log)*np.logaddexp(0,x[1024:1027]+bias);ge=6e-6*np.abs(g)+2*TINY
    decay=np.exp(g);de=np.maximum(np.exp(g+ge)*(1+2e-6)-decay,decay-np.exp(g-ge)*(1-2e-6))+2*TINY
    beta,be=rounded_error(sigmoid(x[1027:1030]),2e-6*sigmoid(x[1027:1030])+2*TINY)
    packet=np.zeros((3,386),np.float64);bound=np.zeros_like(packet)
    for head in range(3):
        at=slice(256+head*128,384+head*128)
        packet[head]=np.r_[shared,activation[at],decay[head],beta[head]]
        bound[head]=np.r_[se,ae[at],de[head],be[head]]
    return packet,bound,history.copy()


def gated_interval(core_bits,z_bits,gain_bits):
    core=expand(core_bits).astype(np.float64);z=expand(z_bits).astype(np.float64);gain=expand(gain_bits).astype(np.float64)
    lo,hi=normalized_interval(core,np.zeros(128),True);lo,hi=bf(lo),bf(hi)
    lo,hi=product_interval(lo,hi,gain,gain);lo,hi=bf(lo),bf(hi)
    factor=silu(z);error=5e-6*np.abs(factor)+2*TINY
    lo,hi=product_interval(lo,hi,factor-error,factor+error)
    extra=gamma(2)*np.maximum(np.abs(lo),np.abs(hi))+TINY
    return bf(lo-extra),bf(hi+extra)
