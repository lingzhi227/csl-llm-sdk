"""Exact saved-evidence checks; packet correctness and overlap are separate."""
from itertools import permutations

HEADER = 0x00400120
ROUTING = [0x7e20, 0x7e60, 0x7e80, 0x7e40, 0x7e00, 0xa0, 0xa0]
STATS = [2,0,2,2,0,0,0,0,2,0,0,0,0,0,1,2]

def require(value, message):
    if not value: raise ValueError(message)

def matrix(value, rows, cols, name):
    require(isinstance(value,list) and len(value)==rows,name+' rows')
    require(all(isinstance(r,list) and len(r)==cols for r in value),name+' columns')
    require(all(type(x) is int and 0<=x<=0xffffffff for r in value for x in r),name+' u32')

def length(owner, sequence):
    return 8 if owner==1 else 31 if sequence==0 else 26

def word(owner, sequence, offset):
    tag=(owner<<16)|(sequence<<8)
    if offset==0:return 0xa1000000|tag
    if offset==1:return 0xb2000000|tag|length(owner,sequence)
    if 2<=offset<=7:return [0,0xffffffff,0x80000000,0x7fffffff,0x00290000,0x00400128][offset-2]
    return 0x5a000000|tag|offset

def identity(value):
    owner=(value>>16)&255;sequence=(value>>8)&255
    require(owner in (1,2) and sequence in (0,1),'packet identity')
    require(value==word(owner,sequence,0),'complete identity word')
    return owner,sequence

def initialized(state,routing):
    matrix(state,3,48,'initial state');matrix(routing,3,8,'routing')
    for owner,row in enumerate(state):require(row==[1,owner]+[0]*46,'exact initialized cache')
    for row in routing:require(row[:7]==ROUTING and row[7]&4==4,'actual MP routing configuration')

def settled(state):
    matrix(state,3,48,'state')
    for owner,row in enumerate(state):
        if row[0:2]!=[2,owner] or row[4:6]!=[0,0] or row[12:20]!=[1,0,0,0,0,1,0,0]:return False
        if row[22:25]!=([0,4,0] if owner==0 else [0,0,0]):return False
        if row[25:31]!=([4,4,4,0,0,0] if owner==0 else [0,0,0,0,2,2]):return False
        if row[2:4]!=([0,4] if owner==0 else [2,0]):return False
    return True

def stable(baseline,final):
    require(settled(baseline),'complete settled baseline required')
    require(settled(final),'complete final state required')
    require(baseline==final,'whole cached state changed during final window')

def check_protocol(state,observed,tx_proof,rx_banks):
    matrix(state,3,48,'state');matrix(observed,1,124,'observed')
    matrix(tx_proof,2,128,'TX proof');matrix(rx_banks,3,31,'RX banks')
    require(settled(state),'protocol has not settled')
    expected_rx=[0]*31;seen={1:0,2:0};order=[];order_word=0
    for arrival in range(4):
        bank=observed[0][arrival*31:(arrival+1)*31]
        owner,sequence=identity(bank[0]);require(sequence==seen[owner],'source-local ordering')
        seen[owner]+=1;order.append([owner,sequence]);n=length(owner,sequence)
        expected_rx[:n]=[word(owner,sequence,i) for i in range(n)]
        require(bank==expected_rx,'complete RX body and retained suffix')
        order_word|=((owner-1)*2+sequence+1)<<(arrival*4)
    require(seen=={1:2,2:2},'exact four packets')
    require(rx_banks==[expected_rx,[0]*31,[0]*31],'complete final RX banks')
    receiver=[2,0,0,4,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,HEADER+length(*order[-1]),length(*order[-1]),0,4,0,4,4,4,0,0,0,order_word]+[0]*16
    require(state[0]==receiver,'whole receiver state')
    for owner in (1,2):
        frame=[0]*32;expected=[]
        for sequence in range(2):
            source=[word(owner,sequence,i) for i in range(31)]
            n=length(owner,sequence);frame[0]=HEADER+n;frame[1:n+1]=source[:n]
            expected+=source+frame.copy()
        expected+=[7,7]
        require(tx_proof[owner-1]==expected,'complete source/frame banks and lease proofs')
        row=state[owner]
        sender=[2,owner,2,0,0,0,1,1,1,row[9],row[10],row[11],1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,2,2]+STATS+[1]
        require(row==sender,'whole sender state excluding separately checked notices')
    return dict(protocol_passed=True,packets=4,application_words=73,arrival_order=order,source_order_strict=True,complete_RX_suffix_and_TX_banks=True)

def check_overlap(state):
    matrix(state,3,48,'state')
    for owner in (1,2):
        peer=3-owner;row=state[owner]
        require(row[7:9]==[1,1],'first packet started and completed')
        require(row[10:12]==[1,0],'one correct peer-first notification')
        require(row[9]==(peer<<16)|0x107,'both first-packet causal witnesses required')
    return dict(overlap_qualified=True,scope='Overlapping issued-to-software-release source/frame lease intervals only; not simultaneous DMA or router-cycle arbitration.')

def synthetic(order=((1,0),(2,0),(1,1),(2,1))):
    """Independent expected records for bounded host-only checker contradictions."""
    observed=[];rx=[0]*31;order_word=0
    for arrival,(owner,sequence) in enumerate(order):
        n=length(owner,sequence);rx[:n]=[word(owner,sequence,i) for i in range(n)]
        observed.extend(rx);order_word|=((owner-1)*2+sequence+1)<<(arrival*4)
    receiver=[2,0,0,4,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,HEADER+length(*order[-1]),length(*order[-1]),0,4,0,4,4,4,0,0,0,order_word]+[0]*16
    state=[receiver];proof=[]
    for owner in (1,2):
        state.append([2,owner,2,0,0,0,1,1,1,((3-owner)<<16)|0x107,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,2,2]+STATS+[1])
        banks=[];frame=[0]*32
        for sequence in range(2):
            source=[word(owner,sequence,i) for i in range(31)];n=length(owner,sequence)
            frame[0]=HEADER+n;frame[1:n+1]=source[:n];banks+=source+frame.copy()
        proof.append(banks+[7,7])
    return dict(state=state,observed=[observed],tx_proof=proof,rx_banks=[rx,[0]*31,[0]*31])
