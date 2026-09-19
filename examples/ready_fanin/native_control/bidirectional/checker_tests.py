"""Seven legal interleavings and independent evidence contradictions, no SDK."""
from copy import deepcopy
import json
from pathlib import Path
from check_lifecycle import check_protocol,check_overlap,stable,initialized


def fixture(ready_at):
    # Build actual-shaped arrays from a separate declarative packet transcript.
    rows=[]
    for group in (0,1):
        values=[0x3f00+group*128+i for i in range(128)]
        packed=[values[i]|values[i+1]<<16 for i in range(0,128,2)]
        rows.append(packed)
    responses=[]
    for group in (0,1):
        for offset,count in ((0,23),(23,23),(46,18)):
            responses.append([9,1,1,0,0,2,group,offset]+rows[group][offset:offset+count])
    request=[[11,1,1,0,0,2,g,g*128] for g in (0,1)]
    ready=[10,1,1,0,0,3,0,0]
    origin=responses[:ready_at]+[ready]+responses[ready_at:]
    deliveries=[origin,[],request];tx=[request,[ready],responses]
    observed=[];rx_banks=[];proof=[];frames=[];sources=[]
    for pe in range(3):
        current=[0]*31;archive=[]
        for data in deliveries[pe]:
            current[:len(data)]=data
            archive+=current.copy()+[((4+pe)<<20)+0x120+len(data)]
        observed.append(archive+[0]*(256-len(archive)));rx_banks.append(current.copy())
        src=[0]*31;wire=[0]*32;archive=[]
        for data in tx[pe]:
            src[:len(data)]=data;wire[0]=(0x600120 if pe==0 else 0x400120)+len(data)
            wire[1:len(data)+1]=data;archive+=src.copy()+wire.copy()+[7]
        proof.append(archive+[0]*(384-len(archive)));frames.append(wire.copy());sources.append(src.copy())
    state=[]
    # Values here describe the actual event order of this host-only fixture.
    # They are not observations from a simulator/device.
    markers=((0xff,0x010111ff),(0,0),(0x100000ff,0x211031ff))
    for pe,(sent,got) in enumerate(((2,7),(1,0),(6,2))):
        s=[0]*64;s[:4]=[2,pe,sent,got];s[4]=markers[pe][0];s[6]=markers[pe][1]
        if pe<2:s[7:12]=[1,1,0x10107 if pe==0 else 0x107,1,0]
        s[12:20]=[1,0,0,0,0,1,0,0]
        if got:s[20:22]=[((4+pe)<<20)+0x120+len(deliveries[pe][-1]),len(deliveries[pe][-1])]
        s[22:31]=[0,got,0,got,got,got,0,sent,sent]
        s[31:47]=[sent,0,sent,sent,0,0,0,0,sent,got,0,*([2,2,2] if pe==0 else [3,3,3] if pe==2 else [0,0,0]),got+1,sent]
        if pe==0:
            s[48]=2;s[52:62]=[1,1,2,1,0,0,1,64,1,0];s[62]=8
        elif pe==1:s[47]=1;s[54:56]=[1,1];s[62]=8
        else:s[49:51]=[2,2];s[52]=1;s[54]=2;s[59:64]=[64,1,18,26,1]
        state.append(s)
    return dict(state=state,observed=observed,tx_proof=proof,rx_banks=rx_banks,
                live_frames=frames,live_sources=sources,row_archive=[rows[0]+rows[1],[0]*128,rows[0]+rows[1]])


def run():
    positives=0;negatives=0
    def rejects(fn):
        nonlocal negatives
        try:fn()
        except ValueError:negatives+=1
        else:raise AssertionError('Contradictory fixture accepted')
    for at in range(7):
        f=fixture(at);report=check_protocol(**f);assert report['READY_position']==at
        assert report['local_RX_during_issued_unfinished_TX']==[0,0,0]
        assert report['local_RX_TX_overlap_observed']==[False,False,False]
        check_overlap(f['state']);stable(f['state'],deepcopy(f['state']));positives+=1
        # Every position/role tests untouched slots, current banks and archives.
        for field,pe,index in [('observed',0,250),('observed',1,0),('observed',2,70),
                               ('tx_proof',0,130),('tx_proof',1,70),('tx_proof',2,63),
                               ('row_archive',0,80),('row_archive',2,80),('row_archive',1,0),
                               ('live_frames',0,1),('live_sources',2,28),('rx_banks',0,30),
                               ('state',0,4),('state',0,6),('state',2,4),('state',2,6),
                               ('state',0,54),('state',2,54),('state',0,18)]:
            broken=deepcopy(f);broken[field][pe][index]^=1
            rejects(lambda broken=broken:check_protocol(**broken))
        # Every full RX word/header and TX source/frame/proof is required.
        for field,pe,count in [('observed',0,224),('observed',2,64),
                               ('tx_proof',0,128),('tx_proof',1,64),('tx_proof',2,384)]:
            for index in range(count):
                broken=deepcopy(f);broken[field][pe][index]^=1
                rejects(lambda broken=broken:check_protocol(**broken))
        # In particular group1 cannot erase/check only prefix after group0's
        # short final fragment; both source and frame unused suffix are exact.
        for index in (2*64+30,2*64+31+31,3*64+8,5*64+30,5*64+31+31):
            broken=deepcopy(f);broken['tx_proof'][2][index]^=0x80000000
            rejects(lambda broken=broken:check_protocol(**broken))
        for pe in (0,1):
            for index in (7,8,9,10,11):
                broken=deepcopy(f['state']);broken[pe][index]^=1
                rejects(lambda broken=broken:check_overlap(broken))
        for pe in range(3):
            for index in range(64):
                broken=deepcopy(f['state']);broken[pe][index]^=1
                rejects(lambda broken=broken:stable(f['state'],broken))
    f=fixture(0);f['state'][0][41]=1
    report=check_protocol(**f)
    assert report['local_RX_during_issued_unfinished_TX']==[1,0,0]
    assert report['local_RX_TX_overlap_observed']==[True,False,False]
    for pe,index in [(0,42),(2,42),(0,44),(2,44)]:
        f=fixture(0);f['state'][pe][index]=0
        rejects(lambda f=f:check_protocol(**f))
    f=fixture(0);f['state'][2][43]=2
    rejects(lambda:check_protocol(**f))
    initialized([[1,x]+[0]*62 for x in range(3)],
                [[0x7e20,0x7e60,0x7e80,0x7e40,0x7e00,0xa0,0xa0,4] for _ in range(3)])
    # Scalar type/shape attacks cannot be accepted through Python truthiness.
    f=fixture(0);f['state'][0][0]=True;rejects(lambda:check_protocol(**f))
    f=fixture(0);f['row_archive'][0].pop();rejects(lambda:check_protocol(**f))
    return dict(passed=True,legal_interleavings=positives,contradictions=negatives,SDK_imports=0,SDK_invocations=0)


if __name__=='__main__':
    print(json.dumps(run(),sort_keys=True))
