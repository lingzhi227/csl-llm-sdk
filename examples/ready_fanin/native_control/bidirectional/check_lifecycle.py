"""SDK-free, complete saved-bank and actual lifecycle observation checks."""
TX=(2,1,6)
RX=(7,0,2)
ROUTING=[0x7e20,0x7e60,0x7e80,0x7e40,0x7e00,0xa0,0xa0]


def require(value,message):
    if not value:raise ValueError(message)


def matrix(value,rows,cols,name):
    require(isinstance(value,list) and len(value)==rows,name+' rows')
    require(all(isinstance(row,list) and len(row)==cols for row in value),name+' columns')
    require(all(type(v) is int and 0<=v<=0xffffffff for row in value for v in row),name+' u32')


def network(destination,length):
    return ((4+destination)<<20)|0x120|length


def header(kind,phase,group,detail=0):
    return [kind,1,1,0,0,phase,group,detail]


def row_values(group):
    return [0x3f00+group*128+i for i in range(128)]


def packed_row(group):
    values=row_values(group)
    return [values[2*i]|values[2*i+1]<<16 for i in range(64)]


def fragments():
    return [header(9,2,g,o)+packed_row(g)[o:o+n]
            for g in range(2) for o,n in ((0,23),(23,23),(46,18))]


def requests():
    return [header(11,2,g,g*128) for g in range(2)]


def causal_marker(owner,group):
    # These are required actual event snapshots; never reconstructed from final
    # counts when reading evidence. Group0/1 must each have its own stored word.
    require(owner in (0,2) and group in (0,1),'marker identity')
    completed=group if owner==0 else group*3
    rows_done=group if owner==0 else 0
    peer_rows=group if owner==2 else 0
    received=group+1 if owner==2 else 0
    return 255|(group<<8)|(completed<<12)|(rows_done<<16)|(peer_rows<<20)|(group<<24)|(received<<28)


def initialized(state,routing):
    matrix(state,3,64,'initial state');matrix(routing,3,8,'routing')
    for owner,row in enumerate(state):require(row==[1,owner]+[0]*62,'exact initialized cache')
    for row in routing:require(row[:7]==ROUTING and row[7]&4==4,'actual MP routing')


def settled(state):
    matrix(state,3,64,'state')
    for owner,row in enumerate(state):
        if row[:4]!=[2,owner,TX[owner],RX[owner]]:return False
        if row[5]!=0 or row[12:20]!=[1,0,0,0,0,1,0,0]:return False
        if row[22:31]!=[0,RX[owner],0,RX[owner],RX[owner],RX[owner],0,TX[owner],TX[owner]]:return False
    return state[0][48]==2 and state[0][53:55]==[1,2] and state[2][49:52]==[2,2,0] and state[2][54]==2


def stable(baseline,final):
    require(settled(baseline),'complete settled baseline required')
    require(settled(final),'complete final state required')
    require(baseline==final,'whole cached state changed during final window')


def check_stats(owner,stats):
    tx,rx=TX[owner],RX[owner]
    for index,value in {0:tx,2:tx,3:tx,4:0,7:0,8:tx,9:rx,14:rx+1,15:tx}.items():
        require(stats[index]==value,'actual packet statistic '+str(index))
    require(0<=stats[1]<=tx,'receive-held queued request bound')
    # Each retry/stale activation originates from a send or RX completion;
    # the finite operation count bounds observations without fixing timing.
    require(stats[2]+stats[5]+stats[6]<=tx+rx,'finite retry/stale activations')
    require(0<=stats[10]<=rx,'RX during issued unfinished TX count')
    # Last origin RX follows REQUEST1 issue; last peer RX is REQUEST1,
    # after all3 row0 frames issued and before any row1 frame can be issued.
    starts=(2,0,3)[owner];minimum_completed=(1,0,2)[owner]
    require(stats[11]==starts and stats[13]==starts and minimum_completed<=stats[12]<=starts,
            'latest RX causal TX observations')
    if stats[12]<starts:require(stats[10]>0,'unfinished TX at latest RX requires actual overlap count')
    if rx==0:require(stats[10:14]==[0]*4,'no fabricated RX observations')


def check_protocol(state,observed,tx_proof,rx_banks,live_frames,live_sources,row_archive):
    for name,value,cols in [('state',state,64),('observed',observed,256),('TX proof',tx_proof,384),
                           ('RX banks',rx_banks,31),('live frames',live_frames,32),
                           ('live sources',live_sources,31),('actual row archive',row_archive,128)]:
        matrix(value,3,cols,name)
    require(settled(state),'protocol has not settled')
    origin_payloads=[];row_index=0;ready_position=None
    expected_rows=fragments()
    for arrival in range(7):
        bank=observed[0][arrival*32:(arrival+1)*32]
        if bank[0]==10:
            require(ready_position is None,'READY delivered once')
            expected=header(10,3,0);ready_position=arrival
        else:
            require(row_index<6,'extra row fragment');expected=expected_rows[row_index];row_index+=1
        require(bank[:len(expected)]==expected,'complete ordered origin payload')
        origin_payloads.append(expected)
    require(ready_position is not None and row_index==6,'one READY plus six fragments')
    deliveries=[origin_payloads,[],requests()]
    for owner,packets in enumerate(deliveries):
        bank=[0]*31
        for arrival,payload in enumerate(packets):
            bank[:len(payload)]=payload
            require(observed[owner][arrival*32:(arrival+1)*32]==bank+[network(owner,len(payload))],
                    'complete actual RX bank/header/retained suffix')
        require(observed[owner][len(packets)*32:]==[0]*(256-len(packets)*32),'unused RX slots zero')
        require(rx_banks[owner]==bank,'whole current RX bank')
    transmissions=[requests(),[header(10,3,0)],expected_rows]
    for owner,packets in enumerate(transmissions):
        source=[0]*31;frame=[0]*32
        for index,payload in enumerate(packets):
            # row_packet updates only its transmitted prefix. The final five
            # source words of each26-word fragment retain the preceding source.
            source[:len(payload)]=payload
            destination=2 if owner==0 else 0
            frame[0]=network(destination,len(payload));frame[1:len(payload)+1]=payload
            require(tx_proof[owner][index*64:(index+1)*64]==source+frame+[7],
                    'complete source/frame/suffix and lease proof')
        require(tx_proof[owner][len(packets)*64:]==[0]*(384-len(packets)*64),'unused TX slots zero')
        require(live_frames[owner]==frame,'whole current TX frame')
        require(live_sources[owner]==source,'whole current selected source')
    actual_rows=packed_row(0)+packed_row(1)
    require(row_archive==[actual_rows,[0]*128,actual_rows],'actual assembled and source-release row archives')
    for owner,row in enumerate(state):
        expected=[0]*64
        expected[:4]=[2,owner,TX[owner],RX[owner]]
        if owner in (0,2):expected[4]=causal_marker(owner,0);expected[6]=causal_marker(owner,1)
        if owner in (0,1):expected[7:12]=[1,1,row[9],row[10],row[11]]
        expected[12:20]=[1,0,0,0,0,1,0,0]
        if deliveries[owner]:
            n=len(deliveries[owner][-1]);expected[20:22]=[network(owner,n),n]
        expected[22:31]=[0,RX[owner],0,RX[owner],RX[owner],RX[owner],0,TX[owner],TX[owner]]
        check_stats(owner,row[31:47]);expected[31:47]=row[31:47]
        if owner==0:
            expected[48]=2;expected[52:62]=[1,1,2,1,0,0,1,64,1,0];expected[62]=8
        elif owner==1:
            expected[47]=1;expected[54:56]=[1,1];expected[62]=8
        else:
            expected[49:51]=[2,2];expected[52]=1;expected[54]=2
            expected[59:64]=[64,1,18,26,1]
        require(row==expected,'whole final state / event causal markers')
    return dict(protocol_passed=True,packets=9,application_words=200,READY_position=ready_position,
                actual_assembled_rows=2,actual_source_release_rows=2,causal_start_markers=True,
                complete_RX_suffix_and_TX_banks=True,
                local_RX_during_issued_unfinished_TX=[row[41] for row in state],
                local_RX_TX_overlap_observed=[row[41]>0 for row in state],
                local_RX_TX_scope='Actual packet stats[10]; first-TX notice witnesses alone do not prove local RX/TX overlap')


def check_overlap(state):
    matrix(state,3,64,'state')
    for owner in (0,1):
        other=1-owner;row=state[owner]
        require(row[7:9]==[1,1],'first packet issued and completed')
        require(row[10:12]==[1,0],'one correct peer-first notice')
        require(row[9]==((other<<16)|0x107),'both first-lease causal witnesses required')
    require(state[2][7:12]==[0]*5,'responder has no synthetic notice')
    return dict(overlap_qualified=True,scope='Issued-to-software-release interval overlap only; not simultaneous DMA.')
