"""Strict SDK-free interpretation of actual independent sink records."""
WIDTH=178
EXPECTED={0:161,2:81,**{x:3 for x in range(40,176)}}
CAPACITIES={0:192,2:96,**{x:3 for x in range(40,176)}}
def require(ok,message):
    if not ok:raise ValueError(message)
def header(kind,phase,group,detail=0):return [kind,1,1,0,0,phase,group,detail]
def records(words,capacity):
    require(len(words)==capacity*16,'record bank extent')
    out=[];torn=False
    for i in range(capacity):
        row=[int(v) for v in words[16*i:16*i+16]]
        if row==[0]*16:break
        if row[0]!=2*(i+1) or row[15]!=row[0]:torn=True;break
        require(0<=row[3]<=31 and row[13]==(1<<min(row[3],8))-1,'valid length/mask')
        out.append(row)
    return out,torn

def check(status,origin,peer,producers,*,require_complete=True):
    require(len(status)==WIDTH and all(len(v)==8 for v in status),'sink status rectangle')
    require(len(producers)==136,'producer capture count')
    decoded={};torn=[]
    for x,data in [(0,origin),(2,peer),*((40+i,p) for i,p in enumerate(producers))]:
        st=[int(v) for v in status[x]];cap=CAPACITIES[x]
        require(st[0]==1 and st[6]==cap and st[7]==1,'sink initialized/compute/capacity')
        require(0<=st[1]<=cap and 0<=st[2]<=cap,'bounded sink counts')
        if require_complete:require(st[2]==st[1] and st[3]==0 and st[4]==0,'sink overflow/ordering')
        rr,bad=records(data,cap);decoded[x]=rr
        if bad:torn.append(x)
    errors=[dict(column=x,sequence=r[0]//2,event=r[1],length=r[3],network_header=r[4],first8=r[5:13],reason=r[14]) for x,rr in decoded.items() for r in rr if r[14]]
    summary=dict(scope='One simultaneous 136-producer READY round with eight original three-fragment rows; synthetic protocol fixture, not original neural inference.',complete=False,coherent_records=sum(map(len,decoded.values())),torn_columns=torn,first_reported_error=errors[0] if errors else None,source_snapshots=sum(len(decoded[x]) for x in range(40,176)),original_neural_epochs=0,source_snapshot_scope='Values at enqueue, immediately before SDK call and source-completion callback; not a continuous DMA immutability proof.',sender_append_overflow_observed=False,potential_saturated_columns=[x for x,rr in decoded.items() if len(rr)==CAPACITIES[x]],sink_errors=[dict(column=x,overflow=int(status[x][3]),malformed=int(status[x][4])) for x in EXPECTED if int(status[x][3]) or int(status[x][4])])
    if not require_complete:return summary
    require(not torn and not errors,'torn or error record')
    require(all(len(decoded[x])==EXPECTED[x] and int(status[x][1])==EXPECTED[x] for x in EXPECTED),'complete finite sink record counts')
    for owner in range(136):
        rr=decoded[40+owner]
        require([r[1] for r in rr]==[1,2,3],'enqueue/preSDK/completion order')
        for r in rr:require(r[2]==owner and r[3]==8 and r[4]==0 and r[5:13]==header(10,3,owner),'producer three point READY snapshots')
    ready=[];fragments=[]
    for r in decoded[0][:-1]:
        require(r[1]==4 and r[4]==0x00400120+r[3],'origin exact network header')
        h=r[5:13]
        if h[0]==10:
            require(r[3]==8 and h==header(10,3,h[6]) and 0<=h[6]<136,'strict READY reserved zero')
            ready.append(h[6])
        else:fragments.append((r[3],h))
    require(sorted(ready)==list(range(136)),'all136 READY once')
    expected=[(length,header(9,2,group,offset)) for group in range(8) for length,offset in [(31,0),(31,23),(26,46)]]
    require(fragments==expected,'original31/31/26 receive sequence')
    r=decoded[0][-1];require(r[1]==5 and r[5:8]==[136,8,0],'origin terminal/whole128halfword device comparison')
    p=decoded[2];requests=[];sent=[];request_at={};events_at={}
    for at,r in enumerate(p[:-1]):
        h=r[5:13];group=h[6]
        if r[1]==4:
            require(r[3]==8 and r[4]==0x00600128 and 0<=group<8 and h==header(11,2,group,group*128),'peer request identity')
            require(group not in request_at,'no duplicate pending request')
            request_at[group]=at;requests.append(group)
        else:
            require(r[1] in [1,2,3] and r[4]==0,'peer source event type')
            sent.append((r[1],r[3],h));key=(group,h[7],r[1]);require(key not in events_at,'unique fragment event');events_at[key]=at
    require(requests==list(range(8)),'ordered unique peer requests')
    expected_sent=[(event,length,header(9,2,group,offset)) for group in range(8) for length,offset in [(31,0),(31,23),(26,46)] for event in [1,2,3]]
    require(sent==expected_sent,'peer fragment enqueue/preSDK/completion partial order')
    for group in range(8):
        require(request_at[group]<events_at[group,0,1],'request before new row enqueue')
        if group:
            require(events_at[group-1,46,3]<events_at[group,0,1],'prior source released before new row enqueue')
            require(events_at[group-1,46,2]<request_at[group],'prior final payload issued before next request')
    require(p[-1][1]==5 and p[-1][5:7]==[8,0],'peer terminal')
    summary.update(complete=True,READY_producers=136,concurrent_rounds=1,request_rows=8,origin_records=161,peer_records=81,source_snapshots=408,all_three_source_point_snapshots_equal=True,all_reserved_words_zero=True,normal_stop_separate=True)
    return summary
