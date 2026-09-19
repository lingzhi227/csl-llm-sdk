"""Reject contradictory simulator evidence; these are not device protocol tests."""
from copy import deepcopy
import json
from check_case import check,LENGTHS

def evidence(case):
    n=LENGTHS[case];state=[[0]*30 for _ in range(2)];rx,tx=state
    for row in state:row[0]=2
    rx[28]=tx[29]=0x400120+n
    delivered=int(case<4 or case==6);rx[2]=delivered
    observed=[0]*31
    if delivered:
        rx[4:6]=[n,1];rx[12:14]=[1,3]
        observed[:n]=[0x11000000+i for i in range(n)]
    if case<4:
        events=[[1,1,1,0,0,0],[0,0,0,0,1,1]]
        rx[18]=rx[23]=1;tx[1]=1;tx[7:12]=[1,1,0,0,1];tx[24:28]=[1,1,2,0]
    else:
        seq,error,phase={4:([1,0,0,1,0,0],11,4),5:([1,1,0,1,0,0],12,4),6:([1,1,1,1,0,0],10,4),7:([1,1,0,0,0,0],0,3)}[case]
        events=[seq,[0]*6];rx[3]=rx[21]=error;rx[18]=phase;tx[6]=1
        if case==5:rx[20]=0x11000008
    return [state,events,observed,[0x11000000+i for i in range(2 if case==4 else n)]+[0]*(31-(2 if case==4 else n))]

def main():
    for case in range(8):assert check(case,*evidence(case))['passed']
    mutations=[
      ('early_raw_prefix_corrupt',4,lambda d:d[3].__setitem__(1,0)),
      ('early_normal_callback',4,lambda d:d[1][0].__setitem__(1,1)),
      ('early_delivery',4,lambda d:d[0][0].__setitem__(2,1)),
      ('extra_data_delivery',5,lambda d:d[0][0].__setitem__(2,1)),
      ('duplicate_not_observed',6,lambda d:d[1][0].__setitem__(3,0)),
      ('duplicate_delivered_twice',6,lambda d:d[0][0].__setitem__(2,2)),
      ('missing_tail_delivered',7,lambda d:d[0][0].__setitem__(2,1)),
      ('source_lease_released_early',3,lambda d:d[0][1].__setitem__(25,0)),
      ('source_callback_before_tail',3,lambda d:d[0][1].__setitem__(27,1)),
      ('valid_dual_callback',3,lambda d:d[1][0].__setitem__(3,1)),
      ('payload_word_corrupt',3,lambda d:d[2].__setitem__(30,0)),
      ('unadvertised_extra_word',0,lambda d:d[2].__setitem__(1,123)),
      ('fixed_length_header',3,lambda d:d[0][0].__setitem__(28,0x40011f)),
      ('routing_bank_enabled',3,lambda d:d[0][0].__setitem__(14,4)),
      ('wrong_error_kind',4,lambda d:d[0][0].__setitem__(3,12)),
    ]
    for name,case,mutate in mutations:
        data=deepcopy(evidence(case));mutate(data)
        try:check(case,*data)
        except ValueError:pass
        else:raise AssertionError(name)
    print(json.dumps(dict(passed=True,synthetic_positive=8,contradictory_evidence_rejected=len(mutations),SDK_invocations=0,hardware_invocations=0)))

if __name__=='__main__':main()
