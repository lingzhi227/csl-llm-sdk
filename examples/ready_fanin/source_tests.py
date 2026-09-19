"""Exercise the real saved-record validator with valid and contradictory evidence."""
from copy import deepcopy
from pathlib import Path
import ast,json
import check_capture as c

def rec(seq,event,h,length=8,network=0,subject=0):return [2*seq,event,subject,length,network,*h,255,0,2*seq]
def pad(rows,cap):return sum(rows,[])+[0]*(16*(cap-len(rows)))
def valid():
    status=[[1,0,0,0,0,1,1,1] for _ in range(178)]
    for x in c.EXPECTED:status[x]=[1,c.EXPECTED[x],c.EXPECTED[x],0,0,1,c.CAPACITIES[x],1]
    producers=[pad([rec(i+1,i+1,c.header(10,3,owner),subject=owner) for i in range(3)],3) for owner in range(136)]
    origin=[]
    for owner in range(136):origin.append(rec(len(origin)+1,4,c.header(10,3,owner),network=0x400108,subject=owner))
    peer=[]
    for group in range(8):
        peer.append(rec(len(peer)+1,4,c.header(11,2,group,128*group),network=0x600108,subject=group))
        for n,offset in [(31,0),(31,23),(26,46)]:
            h=c.header(9,2,group,offset);origin.append(rec(len(origin)+1,4,h,length=n,network=0x400100+n,subject=group))
            for event in [1,2,3]:peer.append(rec(len(peer)+1,event,h,length=n,subject=group))
    origin.append(rec(len(origin)+1,5,[136,8,0,0,0,0,0,0]));peer.append(rec(len(peer)+1,5,[8,0,0,0,0,0,0,0]))
    return [status,pad(origin,192),pad(peer,96),producers]
def reorder_peer(data,old,new):
    rows=[data[2][i:i+16] for i in range(0,81*16,16)];r=rows.pop(old);rows.insert(new,r)
    for i,r in enumerate(rows):r[0]=r[15]=2*(i+1)
    data[2]=pad(rows,96)
def main():
    data=valid();assert c.check(*data)['complete'];race=deepcopy(data);reorder_peer(race,10,9);assert c.check(*race)['complete'];cases=[]
    changes=[('source_reserved',lambda d:d[3][122].__setitem__(12,0x400108)),('source_preSDK',lambda d:d[3][122].__setitem__(28,9)),('missing_producer',lambda d:d[3].pop()),('origin_reserved',lambda d:d[1].__setitem__(12,0x400108)),('network_header',lambda d:d[1].__setitem__(4,0x400107)),('duplicate_owner',lambda d:d[1].__setitem__(27,0)),('sink_overflow',lambda d:d[0][0].__setitem__(3,1)),('record_torn',lambda d:d[1].__setitem__(15,0)),('peer_request',lambda d:d[2].__setitem__(12,1)),('missing_terminal',lambda d:d[0][0].__setitem__(1,160)),('enqueue_before_release',lambda d:reorder_peer(d,11,9)),('enqueue_before_request',lambda d:reorder_peer(d,11,10)),('duplicate_pending_request',lambda d:d[2].__setitem__(10*16+11,0))]
    for name,mutate in changes:
        d=deepcopy(data);mutate(d)
        try:c.check(*d)
        except (ValueError,IndexError):cases.append(name)
        else:raise AssertionError(name)
    d=deepcopy(data);d[1][12]=0x400108;d[1][14]=4
    assert c.check(*d,require_complete=False)['first_reported_error']['reason']==4
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
    base=(Path(__file__).parent/'production-packets.csl').read_text();current=(Path(__file__).parent/'packets.csl').read_text()
    expected=base.replace('param on_sent:fn()void;','param before_sdk:fn([*]u32,u16)void;\nparam on_sent:fn()void;').replace(' mp.send_message(pending_x,pending_y,pending_source,pending_length,.{.activate=sent_task_id});',' before_sdk(pending_source,pending_length);\n mp.send_message(pending_x,pending_y,pending_source,pending_length,.{.activate=sent_task_id});')
    assert current==expected
    print(json.dumps(dict(passed=True,positive=2,negative=len(cases),partial_first_error=1,transport_delta_exact=True,SDK_invocations=0,hardware_invocations=0)))
if __name__=='__main__':main()
