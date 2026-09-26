"""Shared coordinates for the complete 24-layer resident backend under construction.

These are placement definitions, not evidence of a compiled full-model program.
The host's fixed phase schedule never carries a neural intermediate.
"""
from dataclasses import dataclass

WIDTH=750
HEIGHT=1160
LAYERS=24
HIDDEN=2880
VOCAB=201088
VOCAB_TILES=1571
LAYER_WIDTH=27
EMBEDDING_X=4
HEAD_X=694
HEAD_CONTROLLER=(736,0)
CONTEXT=96

def layer_x(layer):
    if not 0<=layer<LAYERS:raise ValueError('Layer outside the original model')
    return 46+LAYER_WIDTH*layer

def vocabulary_tile(row_tile,k_tile,head=False):
    if not 0<=row_tile<VOCAB_TILES or not 0<=k_tile<30:raise ValueError('Vocabulary tile out of range')
    return ((HEAD_X if head else EMBEDDING_X)+row_tile//38,30*(row_tile%38)+k_tile)

def qkv_tile(layer,row_tile,k_tile):
    if not 0<=row_tile<40 or not 0<=k_tile<30:raise ValueError('QKV tile out of range')
    return (layer_x(layer)+row_tile//38,30*(row_tile%38)+k_tile)

def o_tile(layer,row_tile,k_tile):
    if not 0<=row_tile<23 or not 0<=k_tile<43:raise ValueError('O tile out of range')
    return (layer_x(layer)+1,64+43*row_tile+k_tile)

def router_tile(layer,k_tile):
    if not 0<=k_tile<30:raise ValueError('Router tile out of range')
    return (layer_x(layer)+1,1054+k_tile)

def expert_tile(layer,expert,part,row_tile,k_tile):
    rows,start=(36,4) if part=='gate_up_proj' else (18,15) if part=='down_proj' else (0,0)
    if not 0<=expert<32 or not 0<=row_tile<rows or not 0<=k_tile<10:raise ValueError('Expert tile out of range')
    return (layer_x(layer)+start+k_tile,36*expert+row_tile)

def controller(layer,kind):
    offset,y={'attention':(2,0),'moe':(3,1159),'join':(26,0)}[kind]
    return (layer_x(layer)+offset,y)

PREFIX_NODES=tuple(sorted({0,*range(8),*(29+30*r for r in range(38)),*(106+43*r for r in range(23))}))

@dataclass(frozen=True)
class Role:
    kind:str
    layer:int=-1
    row:int=-1
    contraction:int=-1
    expert:int=-1

def role_at(x,y):
    if not 0<=x<WIDTH or not 0<=y<HEIGHT:raise ValueError('Outside application')
    if (x,y)==(0,0):return Role('west_token')
    if (x,y)==(749,0):return Role('east_token')
    for start,kind in ((EMBEDDING_X,'embedding'),(HEAD_X,'head')):
        if start<=x<start+42 and y<1140:
            row=(x-start)*38+y//30
            if row<VOCAB_TILES:return Role(kind,row=row,contraction=y%30)
            return Role(kind+'_relay',row=row,contraction=y%30)
    if x==HEAD_CONTROLLER[0]:
        if y==0:return Role('head_controller')
        if y<1140 and y%30==29:return Role('head_collector',row=y//30)
    if 46<=x<694:
        layer=(x-46)//LAYER_WIDTH;local=(x-46)%LAYER_WIDTH
        if local==0 and y<1140:return Role('qkv',layer,row=y//30,contraction=y%30)
        if local==1:
            if y<60:return Role('qkv',layer,row=38+y//30,contraction=y%30)
            if 64<=y<1053:return Role('o',layer,row=(y-64)//43,contraction=(y-64)%43)
            if 1054<=y<1084:return Role('router',layer,row=0,contraction=y-1054)
        if local==2:
            if y==0:return Role('attention_controller',layer)
            if y in PREFIX_NODES:return Role('prefix_collector',layer,row=PREFIX_NODES.index(y))
        if local==3:
            if y<8:return Role('kv',layer,row=y)
            if y==1159:return Role('moe_controller',layer)
        if y<1152:
            expert,row=divmod(y,36)
            if 4<=local<=13:return Role('gate',layer,row,local-4,expert)
            if local==14:return Role('expert_middle' if row==0 else 'gate_collector',layer,row,expert=expert)
            if 15<=local<=24 and row<18:return Role('down',layer,row,local-15,expert)
            if local==25 and row<18:return Role('expert_output' if row==0 else 'down_collector',layer,row,expert=expert)
            if local==26 and row==0:return Role('global_join' if expert==0 else 'expert_collector',layer,row,expert=expert)
        return Role('relay',layer)
    return Role('relay')

def qkv_tensor_row(row_tile):
    if row_tile<32:return 'q_proj',row_tile*128
    if row_tile<36:return 'k_proj',(row_tile-32)*128
    if row_tile<40:return 'v_proj',(row_tile-36)*128
    raise ValueError('QKV row outside model')
