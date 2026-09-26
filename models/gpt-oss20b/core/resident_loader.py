"""Bounded original-weight packing for the group-aligned resident placement.

No checkpoint dequantization or model arithmetic. Each yielded host buffer is
at most 1 MiB. MXFP4 nibbles/scales and all BF16 bits remain unchanged.
"""
from collections import Counter
from dataclasses import dataclass
import math
import numpy as np
import resident_geometry as geo

@dataclass
class Transfer:
    symbol:str
    x:int
    y:int
    width:int
    height:int
    count:int
    data:np.ndarray
    tensor:str
    source_bytes:int

    def validate(self):
        assert self.data.dtype==np.uint32 and self.data.flags.c_contiguous
        assert self.data.size==self.width*self.height*self.count
        assert self.data.nbytes<=1<<20
        assert 0<=self.x<self.x+self.width<=geo.WIDTH
        assert 0<=self.y<self.y+self.height<=geo.HEIGHT
        assert np.all(self.data<=65535)
        return self

def transfer(symbol,xy,width,height,count,data,tensor,source_bytes):
    return Transfer(symbol,*xy,width,height,count,np.ascontiguousarray(data,dtype=np.uint32).reshape(-1),tensor,source_bytes).validate()

def dense_weights(checkpoint,name,coordinate,row_block=128):
    raw=checkpoint.tensor(name);rows,columns=raw.shape
    assert raw.dtype==np.uint16
    row_tiles=math.ceil(rows/row_block);k_tiles=math.ceil(columns/96)
    for row in range(row_tiles):
        first=row*row_block;valid_rows=min(row_block,rows-first)
        for k in range(0,k_tiles,15):
            height=min(15,k_tiles-k);valid_columns=min(height*96,columns-k*96)
            block=np.zeros((row_block,height*96),np.uint16)
            block[:valid_rows,:valid_columns]=raw[first:first+valid_rows,k*96:k*96+valid_columns]
            tile=np.ascontiguousarray(block.reshape(row_block,height,96).transpose(1,2,0))
            xy=coordinate(row,k)
            assert coordinate(row,k+height-1)==(xy[0],xy[1]+height-1)
            yield transfer('weights',xy,1,height,row_block*96,tile,name,valid_rows*valid_columns*2)

def dense_bias(checkpoint,name,coordinate,k_tiles,row_block=128):
    raw=checkpoint.tensor(name);assert raw.dtype==np.uint16 and raw.ndim==1
    for row in range(math.ceil(len(raw)/row_block)):
        values=raw[row*row_block:(row+1)*row_block];padded=np.zeros(row_block,np.uint16);padded[:len(values)]=values
        yield transfer('bias',coordinate(row,k_tiles-1),1,1,row_block,padded,name,values.nbytes)

def vector(checkpoint,name,symbol,xy):
    raw=checkpoint.tensor(name);assert raw.dtype==np.uint16 and raw.ndim==1
    yield transfer(symbol,xy,1,1,len(raw),raw,name,raw.nbytes)

def experts(checkpoint,layer):
    for part,rows in [('gate_up_proj',36),('down_proj',18)]:
        prefix=f'model.layers.{layer}.mlp.experts.{part}'
        blocks=checkpoint.tensor(prefix+'_blocks');scales=checkpoint.tensor(prefix+'_scales');bias=checkpoint.tensor(prefix+'_bias')
        assert blocks.shape==(32,rows*160,90,16) and scales.shape==(32,rows*160,90) and bias.shape==(32,rows*160)
        for expert in range(32):
            for row in range(0,rows,2):
                height=min(2,rows-row);first=row*160;last=(row+height)*160
                b=np.ascontiguousarray(blocks[expert,first:last].reshape(height,160,10,9,16).transpose(0,2,1,3,4))
                s=np.ascontiguousarray(scales[expert,first:last].reshape(height,160,10,9).transpose(0,2,1,3))
                xy=geo.expert_tile(layer,expert,part,row,0)
                yield transfer('weights',xy,10,height,11520,b.reshape(-1).view('<u2'),prefix+'_blocks',b.nbytes)
                yield transfer('scales',xy,10,height,720,s.reshape(-1).view('<u2'),prefix+'_scales',s.nbytes)
                values=bias[expert,first:last]
                yield transfer('bias',geo.expert_tile(layer,expert,part,row,9),1,height,160,values,prefix+'_bias',values.nbytes)

def iter_weights(checkpoint):
    yield from dense_weights(checkpoint,'model.embed_tokens.weight',lambda row,k:geo.vocabulary_tile(row,k))
    for layer in range(geo.LAYERS):
        prefix=f'model.layers.{layer}.'
        for projection,offset in [('q_proj',0),('k_proj',32),('v_proj',36)]:
            coordinate=lambda row,k,l=layer,o=offset:geo.qkv_tile(l,row+o,k)
            yield from dense_weights(checkpoint,prefix+'self_attn.'+projection+'.weight',coordinate)
            yield from dense_bias(checkpoint,prefix+'self_attn.'+projection+'.bias',coordinate,30)
        coordinate=lambda row,k,l=layer:geo.o_tile(l,row,k)
        yield from dense_weights(checkpoint,prefix+'self_attn.o_proj.weight',coordinate)
        yield from dense_bias(checkpoint,prefix+'self_attn.o_proj.bias',coordinate,43)
        coordinate=lambda row,k,l=layer:geo.router_tile(l,k)
        yield from dense_weights(checkpoint,prefix+'mlp.router.weight',coordinate,row_block=32)
        yield from dense_bias(checkpoint,prefix+'mlp.router.bias',coordinate,30,row_block=32)
        yield from vector(checkpoint,prefix+'input_layernorm.weight','gain',geo.controller(layer,'attention'))
        yield from vector(checkpoint,prefix+'post_attention_layernorm.weight','gain',geo.controller(layer,'moe'))
        name=prefix+'self_attn.sinks';sinks=checkpoint.tensor(name);assert sinks.shape==(64,) and sinks.dtype==np.uint16
        for head in range(8):
            values=sinks[head*8:(head+1)*8]
            yield transfer('sinks',(geo.layer_x(layer)+3,head),1,1,8,values,name,values.nbytes)
        yield from experts(checkpoint,layer)
    yield from vector(checkpoint,'model.norm.weight','gain',geo.HEAD_CONTROLLER)
    yield from dense_weights(checkpoint,'lm_head.weight',lambda row,k:geo.vocabulary_tile(row,k,head=True))

def audit(checkpoint,consume=None):
    coverage=Counter();counts=Counter();host_bytes=0;largest=0;operations=0
    for item in iter_weights(checkpoint):
        coverage[item.tensor]+=item.source_bytes;counts[item.symbol]+=1
        host_bytes+=item.data.nbytes;largest=max(largest,item.data.nbytes);operations+=1
        if consume is not None:consume(item)
    expected={name:record['data_offsets'][1]-record['data_offsets'][0]
              for _,header in checkpoint.headers.values() for name,record in header.items() if name!='__metadata__'}
    assert set(expected)==set(checkpoint.index)
    if dict(coverage)!=expected:raise ValueError('Not all original tensor bytes were covered exactly once')
    return dict(passed=True,scope='packing and logical byte coverage; no full-model CSL execution',tensors=len(coverage),
                original_bytes=sum(coverage.values()),host_slot_bytes=host_bytes,largest_buffer_bytes=largest,
                transfers=operations,by_symbol=dict(counts),weight_residency='requires successful device loading; not proven by this packing audit')
