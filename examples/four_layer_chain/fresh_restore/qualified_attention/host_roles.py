"""Actual vertical Layer3 role reconstruction from frozen physical plans."""
from collections import Counter
def roles(plan,transport):
    w,h=plan['application'];assert (w,h)==(29,1160)
    result={(x,y):dict(role=0,ordinal=0,participants=54,matrix_kind=0,matrix_group=0,head_id=0) for y in range(h) for x in range(w)}
    def put(point,role,**extra):
        point=tuple(point);assert point in result and result[point]['role']==0;result[point].update(role=role,**extra)
    for name,role in [('origin',6),('input_norm',2),('post_norm',3),('observer',7)]:put(plan[name],role)
    for key,role in [('attention_heads',4),('head_observers',8),('mlp_owners',5)]:
        for i,p in enumerate(plan[key]):put(p,role,head_id=i if role in (4,8) else 0)
    for s in plan['stripes']:
        for i in range(s['columns']):put((s['x'],s['y']+i),1,ordinal=i,participants=s['columns'],matrix_kind=s['packet_kind'] if i==0 else 0,matrix_group=s['group'] if i==0 and s['packet_kind'] in (2,3) else 0)
    for n in transport['nodes']:put((n['x'],n['y']),9,mask=n['mask'],spine=n['spine'])
    assert Counter(p['role'] for p in result.values())=={0:2237,1:30576,2:1,3:1,4:24,5:136,6:1,7:1,8:24,9:639}
    return result
