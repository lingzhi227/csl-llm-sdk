"""Declared complete Layer0 roles and banks; no SDK or model payload imports."""
from collections import Counter
import re


def roles(plan, transport):
    width,height=plan['application']
    assert [width,height]==[30,1160]
    result={(x,y):dict(role=0,ordinal=0,participants=54,matrix_kind=0) for y in range(height) for x in range(width)}
    def set_role(xy,role,**extra):
        point=tuple(xy);assert point in result and result[point]['role']==0
        result[point].update(role=role,**extra)
    for name,role in [('input_norm',2),('post_norm',3),('origin',6),('observer',7)]:set_role(plan['support'][name],role)
    for i,xy in enumerate(plan['support']['MLP_owners']):set_role(xy,5,index=i)
    for i,c in enumerate(plan['convolution']):set_role(c['owner'],8,index=i,vector_kind=0 if i<16 else 1 if i<32 else 2)
    for i,h in enumerate(plan['heads']):
        set_role(h['owner'],4,index=i)
        for j,s in enumerate(h['state_shards']):set_role(s['owner'],9,index=4*i+j)
    for s in plan['stripes']:
        for ordinal in range(s['columns']):set_role((s['x'],s['y']+ordinal),1,ordinal=ordinal,participants=s['columns'],matrix_kind=s['kind'])
    for n in transport['nodes']:set_role((n['x'],n['y']),10,mask=n['mask'],spine=n['spine'])
    counts=Counter(v['role'] for v in result.values())
    assert counts=={0:1822,1:31548,2:1,3:1,4:48,5:136,6:1,7:1,8:80,9:192,10:970}
    return result

