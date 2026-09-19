"""Original complete role map, independent of selected-program demotion."""
from collections import Counter

def roles(plan):
    result={(x+4,y+1):dict(role=0,ordinal=0,participants=54,matrix_kind=0,matrix_group=0,head_id=0)
            for y in range(45) for x in range(750)}
    for x,role in [(0,6),(1,2),(2,3),(748,7)]:result[x+4,1]['role']=role
    for x in range(16,176):
        result[x+4,1]['role']=4 if x<40 else 5
        if x<40:result[x+4,1]['head_id']=x-16
    for y in range(1,25):result[752,y+1].update(role=8,head_id=24-y)
    matrix=set()
    for stripe in plan['stripes']:
        x,y,n,kind,group=(stripe[k] for k in ('x','y','columns','packet_kind','group'))
        for ordinal in range(n):
            point=x+ordinal+4,y+1
            if point not in result or result[point]['role']!=0 or point in matrix:
                raise ValueError('Original stripes must be disjoint and within the application')
            matrix.add(point)
            result[point].update(role=1,ordinal=ordinal,participants=n,
                matrix_kind=kind if ordinal==0 else 0,
                matrix_group=group if ordinal==0 and kind in (2,3) else 0)
    counts=Counter(v['role'] for v in result.values())
    if counts!={0:2986,1:30576,2:1,3:1,4:24,5:136,6:1,7:1,8:24}:
        raise ValueError('Exact full original role counts')
    roots=[v for v in result.values() if v['role']==1 and v['matrix_kind'] in (2,3)]
    if len(roots)!=16 or {(v['matrix_kind'],v['matrix_group']) for v in roots}!={(kind,group) for kind in (2,3) for group in range(8)}:
        raise ValueError('All sixteen native source roots')
    return result
