"""Exact two-packet contents, ordering, leases and native callback evidence."""
NAMES=['consecutive31then8']
LENGTHS=[31,8]

def word(index,i):
    first=[0x00000000,0xffffffff,0x80000000,0x7fffffff]
    second=[0xffffffff,0x00000000,0x55555555,0xaaaaaaaa,0x80000001,0x7ffffffe,0x00008000,0xffff7fff]
    return (first[i] if i<4 else 0xa5a50000|i) if index==0 else (second[i] if i<8 else 0x5a5a0000|i)

def check(state,events,observed,received_buffer,tx_evidence,rx_evidence,transmitted_frame,source_payload,tail_normal_events,consumption_events):
    def require(ok,label):
        if not ok:raise ValueError(label)
    for name,value,n in [('state',state,30),('events',events,6),('observed',observed,62),('received_buffer',received_buffer,31),('tx_evidence',tx_evidence,24),('rx_evidence',rx_evidence,12),('transmitted_frame',transmitted_frame,32),('source_payload',source_payload,31),('tail_normal_events',tail_normal_events,1),('consumption_events',consumption_events,2)]:
        require(len(value)==2 and all(len(row)==n for row in value),name+' complete two-PE extent')
        require(all(type(v) is int and 0<=v<2**32 for row in value for v in row),name+' complete u32')
    rx,tx=state
    require(all(s[0]==2 and s[14]==4 and s[15]==0 for s in state),'Initialized real MP bankA and no harness errors')
    require(events==[[2,2,2,0,0,0],[0,0,0,0,2,2]],'Exact two original callback sequences')
    require(tail_normal_events==[[0],[0]],'No ordinary tail callback')
    require(consumption_events==[[2,0],[0,0]],'Exactly two native control consumptions and no ordinary guard')
    require(rx[1:4]==[0,2,0] and tx[1:4]==[2,0,0],'Exactly two deliveries and two source completions')
    require(all(s[16:22]==[0,0,1,0,0,0] and s[23]==1 for s in state),'Both final leases released, no fault and header rearmed')
    require(rx[4:6]==[8,1] and rx[12:14]==[1,3] and rx[28]==0x00400128,'Second delivery has correct length/header and held phase3 lease')
    require(tx[7:12]==[1,1,0,0,2] and tx[24:28]==[2,1,2,1] and tx[29]==0x00400128,'Both TX completions preserve original release semantics')
    require(rx[22]==0 and tx[22]==1,'Only one immediate source transition')
    first=[word(0,i) for i in range(31)];second=[word(1,i) for i in range(8)]
    require(observed==[first+second+[0]*23,[0]*62],'Both ordered complete delivered snapshots and untouched suffix')
    require(received_buffer==[second+first[8:],[0]*31],'Second RX prefix and first packet retained suffix')
    require(transmitted_frame==[[0]*32,[0x00400128]+second+first[8:]],'Entire final TX frame with retained suffix')
    require(source_payload==[first,[word(1,i) for i in range(31)]],'Entire sender source replaced after first completion')
    expected_tx=[];expected_rx=[]
    for index,length in enumerate(LENGTHS):
        expected_tx += [length,1,1,1,1,2,index,1,0,0,1,0]
        expected_rx += [length,1,3,0x00400120+length,1,index]
    require(tx_evidence==[[0]*24,expected_tx],'Per-packet source/immutable frame and pre-tail/release proof')
    require(rx_evidence==[expected_rx,[0]*12],'Per-packet length/order/header/phase3 receive lease proof')
    return dict(passed=True,lengths=LENGTHS,deliveries=2,source_completions=2,callback_events=events,native_consumption=consumption_events,full_u32_preserved=True,hardware=False,message_passing_fabric_test=True)


def split_snapshot(words):
    if len(words)!=2 or any(len(row)!=33 for row in words):
        raise ValueError('Complete two-PE33word consecutive snapshot required')
    if any(type(v) is not int or not 0<=v<2**32 for row in words for v in row):
        raise ValueError('Complete snapshot u32 values required')
    return [row[:30] for row in words],[row[30:31] for row in words],[row[31:33] for row in words]

def stable_snapshots(initial,final):
    _,first_tail,first_native=split_snapshot(initial)
    state,tail,native=split_snapshot(final)
    if (first_tail,first_native)!=(tail,native):
        raise ValueError('Consecutive counter trailer changed during final observation window')
    if tail!=[[0],[0]] or native!=[[2,0],[0,0]]:
        raise ValueError('Two actual native consumptions and zero ordinary/tail errors required')
    return state,tail,native
