"""Reject contradictory complete-buffer, lifecycle and first-packet evidence."""
from copy import deepcopy
from itertools import permutations
import json
from check_multisource import *

positive=0
for order in permutations(((1,0),(1,1),(2,0),(2,1))):
    if order.index((1,0))>order.index((1,1)) or order.index((2,0))>order.index((2,1)):continue
    x=synthetic(order);check_protocol(**x);check_overlap(x['state']);stable(x['state'],x['state']);positive+=1
assert positive==6
x=synthetic();negative=0
def reject(fn):
    global negative
    try:fn()
    except (ValueError,IndexError,TypeError):negative+=1
    else:raise AssertionError('Contradictory evidence accepted')

# Every captured full-body/source/frame word matters, including retained suffixes.
for key in ('observed','tx_proof','rx_banks'):
    for row in range(len(x[key])):
        for offset in range(len(x[key][row])):
            bad=deepcopy(x);bad[key][row][offset]^=1
            reject(lambda bad=bad:check_protocol(**bad))
for owner in range(3):
    for offset in range(48):
        if owner and offset in (9,10,11):continue
        bad=deepcopy(x);bad['state'][owner][offset]^=1
        reject(lambda bad=bad:check_protocol(**bad))
for owner in (1,2):
    for field,value in [(9,0),(10,0),(10,2),(11,1),(9,0x107),(9,(owner<<16)|0x107),
                        (9,((3-owner)<<16)|0x117),(9,((3-owner)<<16)|0x127),
                        (9,((3-owner)<<16)|0x10f),(9,((3-owner)<<16)|0x103)]:
        bad=deepcopy(x);bad['state'][owner][field]=value
        check_protocol(**bad) # packet correctness does not silently become overlap proof
        reject(lambda bad=bad:check_overlap(bad['state']))
bad=synthetic(((1,1),(2,0),(1,0),(2,1)));reject(lambda:check_protocol(**bad))
bad=deepcopy(x['state']);bad[0][23]=3;reject(lambda:stable(bad,x['state']))
bad=deepcopy(x['state']);bad[1][31]=3;reject(lambda:stable(x['state'],bad))
initial=[[1,i]+[0]*46 for i in range(3)];routing=[ROUTING+[4] for _ in range(3)]
initialized(initial,routing)
for owner in range(3):
    for offset in range(8):
        bad=deepcopy(routing);bad[owner][offset]^=4 if offset==7 else 1
        reject(lambda bad=bad:initialized(initial,bad))
print(json.dumps(dict(passed=True,positive_interleavings=positive,contradictions_rejected=negative,SDK_invocations=0)))
