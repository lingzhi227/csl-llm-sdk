"""Source-only rejection tests for coherent snapshot counter evidence."""
from copy import deepcopy
import json
from check_snapshot import check_snapshot

def main():
    rejected=[]
    for case in range(8):
        counts=[1 if case==5 else 0,1 if case<4 or case==6 else 0,0]
        raw=[[0]*30+counts,[0]*33]
        state,tail,native=check_snapshot(case,raw,raw)
        assert state==[[0]*30,[0]*30] and tail==[[counts[0]],[0]] and native==[counts[1:],[0,0]]
        # Corrupt each of the three counters on each PE, both consistently and
        # only in the later window. None may be ignored or normalized.
        for pe in range(2):
            for word in range(30,33):
                for stable in [False,True]:
                    final=deepcopy(raw);final[pe][word]^=1
                    initial=deepcopy(final if stable else raw)
                    try:check_snapshot(case,initial,final)
                    except ValueError:rejected.append((case,pe,word,stable))
                    else:raise AssertionError('Changed counter accepted')
    for label,raw in [('missing_word',[[0]*32,[0]*33]),('extra_word',[[0]*34,[0]*33]),('missing_PE',[[0]*33]),('not_u32',[[0]*30+[0,1,2**32],[0]*33])]:
        try:check_snapshot(0,raw,raw)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    assert len(rejected)==100
    print(json.dumps(dict(passed=True,synthetic_positive=8,contradictory_evidence_rejected=len(rejected),SDK_invocations=0)))

if __name__=='__main__':main()
