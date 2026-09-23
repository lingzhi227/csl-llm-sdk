"""Unchanged accepted all-row 96-FMA conditional enclosure."""
import math
import numpy as np
from .common import require
FP32_TINY=2**-126;U32=2**-24;U64=2**-53
def gamma(operations, unit_roundoff):
    if type(operations) is not int or operations < 1 or operations * unit_roundoff >= 1:
        raise ValueError('Invalid reduction operation count')
    product = up(operations * unit_roundoff)
    return up(product / down(1.0 - product))
def down(x):return math.nextafter(x,-math.inf)
def up(x):return math.nextafter(x,math.inf)
def decode(x):return (np.asarray(x,dtype=np.uint32)<<16).view(np.float32)

def local_bounds(weights,inputs):
    """Point-input version of the accepted96-column bound, no scalar casts."""
    require(weights.shape[:-1]==inputs.shape and weights.shape[-2:]==(96,128),'Packed rectangle/input shape')
    wf=decode(weights);raw=inputs.view(np.uint32);exponent=(raw>>23)&255
    require(not np.any(raw&65535) and np.isfinite(inputs).all(),'BF16-exact actual local input')
    require(not np.any((exponent==0)&((raw&0x7fffffff)!=0)),'Unexpected actual subnormal input requires separate FTZ audit')
    we=(weights>>7)&255
    require(not np.any(we==255) and not np.any((we==0)&((weights&32767)!=0)),'Original normal weights or zero padding')
    terms=wf.astype(np.float64)*inputs.astype(np.float64)[...,None]
    require(not np.any((np.abs(terms)<FP32_TINY)&(terms!=0)) and np.max(np.abs(terms))<=float.fromhex('0x1.fffffep127'),
            'Local BF16 products must be normal exact FP32 or zero for the retained source operation-count proof')
    nominal=terms.sum(axis=-2,dtype=np.float64);np.abs(terms,out=terms)
    magnitude=terms.sum(axis=-2,dtype=np.float64);del terms
    gu=lambda x:np.nextafter(x,np.inf);gd=lambda x:np.nextafter(x,-np.inf)
    magnitude_upper=gu(magnitude/down(1.-gamma(96,U64)))
    e64=gu(gamma(96,U64)*magnitude_upper)
    ftz=up(up(192*FP32_TINY)/down(1.-up(192*U32)))
    e32=gu(gu(gamma(192,U32)*magnitude_upper)+ftz)
    low=gd(gd(nominal-e64)-e32);high=gu(gu(nominal+e64)+e32)
    low[magnitude==0]=0;high[magnitude==0]=0
    return low,high
