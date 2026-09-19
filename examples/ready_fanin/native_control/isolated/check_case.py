"""Require actual callback identity, exact payload, phase, and source lease facts."""
LENGTHS=[1,8,26,31,8,8,8,8]
NAMES=['valid1','valid8','valid26','valid31','early_control','extra_data','duplicate_control','missing_control']

def check(case,state,events,observed,received_buffer):
    def require(value,label):
        if not value:raise ValueError(label)
    require(0<=case<8,'known case')
    require(len(state)==2 and all(len(x)==30 for x in state),'state extent')
    require(len(events)==2 and all(len(x)==6 for x in events),'event extent')
    require(len(observed)==len(received_buffer)==31,'entire receive and delivered observation')
    rx,tx=state;er,et=events;n=LENGTHS[case]
    require(all(s[0]==2 and s[14]==0 and s[15]==0 for s in state),'initialized static route/no harness error')
    require(rx[28]==0x00400120+n and tx[29]==0x00400120+n,'actual control-mode wire header and advertised length')
    filled=2 if case==4 else n
    require(received_buffer==[0x11000000+i for i in range(filled)]+[0]*(31-filled),'complete actual RX buffer and untouched suffix')
    delivered=1 if case<4 or case==6 else 0
    require(rx[2]==delivered and tx[2]==0,'exact application delivery count')
    require(observed==([0x11000000+i for i in range(n)]+[0]*(31-n) if delivered else [0]*31),'full payload and untouched suffix')
    if delivered:
        require(rx[4:6]==[n,1] and rx[12:14]==[1,3],'exact length/complete contents/lease during delivery')
    else:require(rx[4:6]==[0,0],'no partial payload delivery')
    if case<4:
        require(er==[1,1,1,0,0,0] and et==[0,0,0,0,1,1],'mutually exclusive normal/control callbacks')
        require(rx[3]==rx[21]==0 and rx[18]==1 and rx[23]==1,'receiver rearms one header')
        require(tx[1]==1 and tx[7:12]==[1,1,0,0,1],'single source completion/immutable frame lease')
        require(tx[24:28]==[1,1,2,0],'lease retained before control transmission')
        require(tx[16:18]==[0,0] and tx[19]==0 and tx[3]==tx[21]==0,'sender release only after tail')
    else:
        require(et==[0]*6 and tx[1]==0 and tx[6]==1 and tx[22]==0,'bounded raw injection finished')
        expected={4:([1,0,0,1,0,0],11,4),5:([1,1,0,1,0,0],12,4),6:([1,1,1,1,0,0],10,4),7:([1,1,0,0,0,0],0,3)}
        expected_events,error,phase=expected[case]
        require(er==expected_events,'exact negative callback counts; extra normal callback is a failure')
        require(rx[3]==rx[21]==error and rx[18]==phase,'actual expected receiver error/phase')
        if case==5:require(rx[20]==0x11000008,'extra ordinary wavelet observed')
        if case==6:require(er[2]==1 and er[3]==1 and rx[3]==10,'second control reached and was rejected during next header')
    return dict(passed=True,case=case,name=NAMES[case],length=n,received=delivered,
                callback_events=events,receiver_error=rx[3],actual_duplicate_observed=case==6,
                hardware=False,message_passing_fabric_test=False)
