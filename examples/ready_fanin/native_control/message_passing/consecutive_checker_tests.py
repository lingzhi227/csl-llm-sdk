"""Offline contradictory-evidence tests; no runtime or compiler imported."""
from copy import deepcopy
import json
from check_consecutive import check,word,stable_snapshots

def fixture():
    first=[word(0,i) for i in range(31)];second=[word(1,i) for i in range(8)]
    s=[[0]*30 for _ in range(2)];rx,tx=s
    for row in s:row[0]=2;row[14]=4;row[18]=1;row[23]=1
    for i,v in {2:2,4:8,5:1,12:1,13:3,28:0x00400128}.items():rx[i]=v
    for i,v in {1:2,7:1,8:1,11:2,22:1,24:2,25:1,26:2,27:1,29:0x00400128}.items():tx[i]=v
    return dict(state=s,events=[[2,2,2,0,0,0],[0,0,0,0,2,2]],observed=[first+second+[0]*23,[0]*62],
        received_buffer=[second+first[8:],[0]*31],
        tx_evidence=[[0]*24,[31,1,1,1,1,2,0,1,0,0,1,0,8,1,1,1,1,2,1,1,0,0,1,0]],
        rx_evidence=[[31,1,3,0x0040013f,1,0,8,1,3,0x00400128,1,1],[0]*12],
        transmitted_frame=[[0]*32,[0x00400128]+second+first[8:]],
        source_payload=[first,[word(1,i) for i in range(31)]],tail_normal_events=[[0],[0]],consumption_events=[[2,0],[0,0]])

def main():
    base=fixture();assert check(**base)['passed'];rejected=[]
    mutations=[]
    for i in range(39):mutations.append(('delivered_word_'+str(i),'observed',0,i))
    for name,pe,index in [('observed',0,39),('observed',1,0),('received_buffer',0,0),('received_buffer',0,8),('received_buffer',1,0),('transmitted_frame',1,0),('transmitted_frame',1,9),('source_payload',1,0),('source_payload',1,30),('state',0,2),('state',1,1),('state',0,18),('state',0,21),('state',0,12),('state',0,13),('state',1,24),('events',0,0),('events',0,2),('events',0,3),('events',1,5),('consumption_events',0,0),('consumption_events',0,1),('consumption_events',1,0),('tail_normal_events',0,0),('tx_evidence',1,4),('tx_evidence',1,8),('tx_evidence',1,18),('tx_evidence',1,22),('rx_evidence',0,1),('rx_evidence',0,2),('rx_evidence',0,11)]:
        mutations.append((name+'_'+str(pe)+'_'+str(index),name,pe,index))
    for label,name,pe,index in mutations:
        bad=deepcopy(base);bad[name][pe][index]^=1
        try:check(**bad)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Contradiction accepted: '+label)
    for label,edit in [('reordered_delivery',lambda b:b['observed'][0].__setitem__(slice(0,2),list(reversed(b['observed'][0][:2])))),('truncated_snapshot',lambda b:b['observed'][0].pop()),('masked_high_bit',lambda b:b['observed'][0].__setitem__(1,0x7fffffff))]:
        bad=deepcopy(base);edit(bad)
        try:check(**bad)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Contradiction accepted: '+label)
    snapshots=[base['state'][0]+[0,2,0],base['state'][1]+[0,0,0]]
    assert stable_snapshots(snapshots,snapshots)==(base['state'],base['tail_normal_events'],base['consumption_events'])
    for pe in range(2):
        for i in range(30,33):
            for stable in [False,True]:
                final=deepcopy(snapshots);final[pe][i]^=1
                initial=final if stable else snapshots
                try:stable_snapshots(initial,final)
                except ValueError:rejected.append(('trailer',pe,i,stable))
                else:raise AssertionError('Consecutive trailer contradiction accepted')
    for raw in [[[0]*32,[0]*33],[[0]*34,[0]*33],[[0]*33],[[0]*30+[0,2,2**32],[0]*33]]:
        try:stable_snapshots(raw,raw)
        except ValueError:rejected.append('bad_snapshot_extent_or_u32')
        else:raise AssertionError('Invalid complete snapshot accepted')
    for pe in range(2):
        bad=deepcopy(base);bad['state'][pe][14]=0
        try:check(**bad)
        except ValueError:rejected.append(('MP_bankA_disabled',pe))
        else:raise AssertionError('Disabled real MP route accepted')
    assert len(rejected)==91
    print(json.dumps(dict(passed=True,synthetic_positive=2,contradictory_evidence_rejected=len(rejected),SDK_invocations=0,hardware_invocations=0)))

if __name__=='__main__':main()
