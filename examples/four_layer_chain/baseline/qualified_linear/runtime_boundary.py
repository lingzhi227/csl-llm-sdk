"""Exact host completion and transport fences for the frozen Layer0 program.

All inspected arrays must already have durable raw receipts. These checks do
not evaluate neural arithmetic and cannot replace the independent raw audit.
"""
from collections import Counter
from .host_roles import roles

def require(ok,message):
    if not ok:raise ValueError(message)

def rows(item,value):
    require(value.shape==(item['height'],item['width'],item['count']),'Exact native boundary shape')
    for iy in range(item['height']):
        for ix in range(item['width']):yield (item['x']+ix,item['y']+iy),value[iy,ix]

def state(item,value,origin,serial,generation,position,phase):
    import numpy as np
    require(phase in ('initialized','prepared','complete','reset'),'Known state boundary')
    expected=np.zeros(value.shape,dtype='<u4')
    expected[:,:,0]=2*serial-(phase=='prepared');expected[:,:,1]=serial
    expected[:,:,4]=generation;expected[:,:,5]=position+(phase in ('prepared','complete'))
    x,y=origin
    if item['x']<=x<item['x']+item['width'] and item['y']<=y<item['y']+item['height']:
        expected[y-item['y'],x-item['x'],2]=4*(serial-(phase=='prepared'))
    require(np.array_equal(value,expected),'Every-PE state/error/serial/generation/position boundary: '+phase)

def observer(value,serial,previous_sequence=0):
    a=[int(x) for x in value.reshape(-1)];require(len(a)==32,'One full observer row')
    coherent=a[0]!=0 and a[0]==a[31] and a[0]%2==0
    if not coherent:return dict(coherent=False,complete=False,sequence=previous_sequence)
    require(a[0]>=previous_sequence and a[0]<=32768,'Monotonic bounded observer sequence')
    require(a[1]==0x4c304f31 and a[2]==1,'Exact Layer0 observer schema')
    require(a[5]==0,'Durable origin sticky fault '+str(a[5]))
    require(0<=a[3]<=serial,'Observer cannot come from a future serial')
    if a[3]!=serial or a[4]!=2*serial:
        return dict(coherent=True,complete=False,sequence=a[0])
    require(a[6:10]==[4*serial,4,17472,17408],'Four complete original input frames')
    require(a[10:19]==[0,450,1,48,1,136,1,752,450],'All producer/recipient/consumer joins')
    require(1<=a[19]<=4 and a[20:24]==[752*serial,752*serial,0,14],'No outstanding origin credit or command')
    require(0<=a[24]<=3*serial and a[25:29]==[4*serial,264*serial,69888*serial,serial-1],
            'Cumulative frame/chunk/word/retirement identity')
    require(a[29:31]==[3,135],'Last original MLP input chunk')
    return dict(coherent=True,complete=True,sequence=a[0],serial=serial,origin_state=a)

class Boundary:
    def __init__(self,plan,transport):
        self.plan=plan;self.roles=roles(plan,transport)
        self.nodes={(n['x'],n['y']):n for n in transport['nodes']}
        self.endpoints={tuple(e['router']):tuple(e['endpoint']) for e in transport['endpoints']}
        self.sent=Counter();self.received=Counter();self.router_rx=Counter();self.router_tx=Counter()
        for t in transport['traffic']:
            self.sent[self.endpoints[tuple(t['source'])]]+=t['packets']
            self.received[self.endpoints[tuple(t['destination'])]]+=t['packets']
        neighbors={xy:{} for xy in self.nodes}
        for e in transport['edges']:
            c,p=tuple(e['child']),tuple(e['parent'])
            cp=1 if self.nodes[c]['spine'] else 2
            pp=(2 if self.nodes[c]['spine'] else 0) if self.nodes[p]['spine'] else 1
            require(cp not in neighbors[c] and pp not in neighbors[p],'Unique tree port neighbor')
            neighbors[c][cp]=p;neighbors[p][pp]=c
        for edge in transport['expected_directed_edge_packets']:
            a,b=tuple(edge['source']),tuple(edge['destination']);count=edge['packets']
            out=[p for p,n in neighbors[a].items() if n==b]
            inp=[p for p,n in neighbors[b].items() if n==a]
            require(len(out)==len(inp)==1,'Actual adjacent directed tree edge')
            self.router_tx[a,out[0]]=count;self.router_rx[b,inp[0]]=count
        for router,endpoint in self.endpoints.items():
            self.router_rx[router,0]=self.sent[endpoint];self.router_tx[router,0]=self.received[endpoint]
        self.producers={self.endpoints[tuple(p['source'])]:p for p in transport['producers']}
        self.receivers={}
        for p in transport['producers']:
            for r in p['recipients']:
                xy=self.endpoints[tuple(r['destination'])]
                self.receivers.setdefault(xy,[]).append((p['id'],r))
        self.offset_order={}
        for index,c in enumerate(plan['convolution']):self.offset_order[tuple(c['owner'])]=[index]
        for index,h in enumerate(plan['heads']):
            self.offset_order[tuple(h['owner'])]=[130+index//3,146+index//3,162+index,80+index,128,129]+[258+4*index+j for j in range(4)]
            for s in h['state_shards']:self.offset_order[tuple(s['owner'])]=[210+index]
        require(self.offset_order.keys()==self.receivers.keys(),'Every receiver has an exact CSL slot order')

    def transport(self,item,value,serial,resets):
        checked=0
        for xy,row in rows(item,value):
            a=[int(v) for v in row]
            if xy in self.nodes:
                node=self.nodes[xy];mask=node['mask'];expected=[0]*32
                expected[1:4]=[mask,*xy]
                for port in range(3):
                    expected[4+port]=serial*self.router_rx[xy,port]
                    expected[7+port]=serial*self.router_tx[xy,port]
                    expected[10+port]=int(bool(mask&(1<<port)))
                    expected[13+port]=expected[16+port]=3
                expected[19]=serial;expected[20]=resets
            else:
                tx,rx=serial*self.sent[xy],serial*self.received[xy]
                expected=[0,tx,tx,tx,rx,rx,1,0]
            require(a==expected,'Complete endpoint/edge counters and idle leases at '+str(xy));checked+=1
        return checked

    def lifecycle(self,item,value,serial):
        name=item['symbol']
        for xy,row in rows(item,value):
            a=[int(v) for v in row];expected=None
            if name=='lane_counters':expected=[serial,serial,serial,4*serial]
            elif name=='linear_source_counters':
                p=self.producers[xy];rs=p['recipients'];n=len(rs)
                expected=[serial,serial*n,serial*sum(r['packets'] for r in rs),serial*n,serial*n,serial*n]
            elif name=='linear_receiver_counters':
                rs=self.receivers[xy];n=len(rs)
                require(1<=a[4]<=4,'Bounded actual receiver ACK queue peak')
                expected=[serial*sum(r['packets'] for _,r in rs),serial*n,serial*n,serial*n,a[4]]
            elif name=='linear_receive_offsets':
                lengths={pid:r['payload_words'] for pid,r in self.receivers[xy]}
                expected=[lengths[pid] for pid in self.offset_order[xy]]
            elif name=='linear_origin_seen':expected=[1]*450
            elif name=='linear_origin_active':expected=[0]*450
            elif name=='linear_origin_next':
                ps=sorted(self.producers.values(),key=lambda p:p['id']);expected=[len(p['recipients']) for p in ps]
            elif name=='frame_counters':
                require(0<=a[0]<=3*serial,'Bounded early frame READY count')
                expected=[a[0],4*serial,264*serial,69888*serial,serial-1]
            if expected is not None:require(a==expected,'Complete actual lifecycle counters: '+name+' '+str(xy))
