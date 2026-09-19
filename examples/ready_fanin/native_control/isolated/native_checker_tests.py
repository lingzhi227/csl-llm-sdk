"""Reject lost, duplicate, wrong-PE and unexpected-data evidence, without SDK calls."""
from check_native import check_native
for case in range(8):
    check_native(case,[[1 if case<4 or case==6 else 0,0],[0,0]])
bad=[(0,[[0,0],[0,0]]),(0,[[2,0],[0,0]]),(0,[[1,1],[0,0]]),
     (0,[[1,0],[1,0]]),(0,[[1,0],[0,1]]),(4,[[1,0],[0,0]]),
     (5,[[1,0],[0,0]]),(7,[[1,0],[0,0]]),(6,[[0,0],[0,0]]),
     (0,None),(0,[[1],[0]]),(8,[[1,0],[0,0]])]
for case,value in bad:
    try:check_native(case,value)
    except ValueError:pass
    else:raise AssertionError((case,value))
print({'passed':True,'synthetic_positive':8,'contradictory_evidence_rejected':len(bad),'SDK_invocations':0})
