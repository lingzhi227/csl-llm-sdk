"""Original Layer3 state and complete physical transport/lifecycle fences."""
from collections import Counter
from .host_roles import roles
from .observer_decode import decode_global,semantic_state,decode_head
from .archive_decode import decode_archive
def require(ok,message):
 if not ok:raise ValueError(message)
def rows(item,value):
 require(value.shape==(item['height'],item['width'],item['count']),'Exact native boundary shape')
 for iy in range(item['height']):
  for ix in range(item['width']):yield (item['x']+ix,item['y']+iy),value[iy,ix]
def state(item,value,plan,serial,generation,position,phase):
 import numpy as np
 require(phase in ('initialized','prepared','complete','reset'),'Known scheduled state boundary')
 expected=np.zeros(value.shape,dtype='<u4');expected[:,:,0]=2*serial-(phase=='prepared');expected[:,:,1]=serial;expected[:,:,4]=generation
 expected[:,:,5]=0 if phase in ('initialized','reset') else position+1
 for xy,a in rows(item,expected):
  if xy==tuple(plan['origin']):a[2]=4*(serial-(phase=='prepared'))
  if xy in {tuple(p) for p in plan['attention_heads']}:a[3]=0 if phase in ('initialized','reset') else position+(phase=='complete')
 require(np.array_equal(value,expected),'All PE serial/reset/position/cache-valid identity: '+phase)
def observer(value,serial,previous_sequence=0):
 a=[int(x) for x in value.reshape(-1)];require(len(a)==32,'One complete origin observer')
 if a[0]==a[31] and a[0]>0 and a[0]%2==0:
  require(a[0]>=previous_sequence and a[0]<=8192,'Monotonic origin snapshot sequence')
  require(a[1]==0x4f425331 and a[2] in (4,5),'Original Layer3 observer schema')
  require(0<=a[3]<=serial,'No future origin serial')
  if a[3]!=serial:return dict(coherent=False,complete=False,sequence=a[0])
 observed=decode_global(a,serial)
 if not observed['coherent']:return dict(coherent=False,complete=False,sequence=previous_sequence)
 semantic_state(observed);require(not observed['protocol_error_latched'],'Origin sticky protocol error')
 return dict(coherent=True,complete=observed['final_origin_snapshot'],sequence=a[0],serial=serial,origin_state=a)
class Boundary:
 def __init__(self,plan,transport):
  self.plan=plan;self.roles=roles(plan,transport);self.nodes={(n['x'],n['y']):n for n in transport['nodes']}
  self.endpoints={tuple(e['router']):tuple(e['endpoint']) for e in transport['endpoints']}
  self.sent=Counter();self.received=Counter();self.router_rx=Counter();self.router_tx=Counter()
  for t in transport['traffic']:
   self.sent[self.endpoints[tuple(t['source'])]]+=t['packets'];self.received[self.endpoints[tuple(t['destination'])]]+=t['packets']
  neighbors={xy:{} for xy in self.nodes}
  for e in transport['edges']:
   c,p=tuple(e['child']),tuple(e['parent']);cp=1 if self.nodes[c]['spine'] else 2;pp=(2 if self.nodes[c]['spine'] else 0) if self.nodes[p]['spine'] else 1
   require(cp not in neighbors[c] and pp not in neighbors[p],'Unique tree port neighbor');neighbors[c][cp]=p;neighbors[p][pp]=c
  for edge in transport['expected_directed_edge_packets']:
   a,b=tuple(edge['source']),tuple(edge['destination']);out=[p for p,n in neighbors[a].items() if n==b];inp=[p for p,n in neighbors[b].items() if n==a]
   require(len(out)==len(inp)==1,'Actual adjacent directed edge');self.router_tx[a,out[0]]=edge['packets'];self.router_rx[b,inp[0]]=edge['packets']
  for r,e in self.endpoints.items():self.router_rx[r,0]=self.sent[e];self.router_tx[r,0]=self.received[e]
 def transport(self,item,value,serial,resets):
  checked=0
  for xy,row in rows(item,value):
   a=[int(v) for v in row]
   if xy in self.nodes:
    mask=self.nodes[xy]['mask'];expected=[0]*32;expected[1:4]=[mask,*xy]
    for port in range(3):
     expected[4+port]=serial*self.router_rx[xy,port];expected[7+port]=serial*self.router_tx[xy,port]
     expected[10+port]=int(bool(mask&(1<<port)));expected[13+port]=expected[16+port]=3
    expected[19]=serial;expected[20]=resets
   else:
    tx,rx=serial*self.sent[xy],serial*self.received[xy];expected=[0,tx,tx,tx,rx,rx,1,0]
   require(a==expected,'Complete endpoint/directed-edge counters and idle leases at '+str(xy));checked+=1
  return checked
 def lifecycle(self,item,value,serial):
  name=item['symbol']
  for xy,row in rows(item,value):
   a=[int(v) for v in row];p=self.roles[xy];expected=None;r=p['role']
   if name=='lane_counters':expected=[serial,serial,serial,4*serial]
   elif name=='frame_counters':
    require(0<=a[0]<=3*serial,'Bounded early frame ready count');expected=[a[0],4*serial,264*serial,69888*serial,serial-1]
   elif name=='native_status':
    expected=[0]*16
    if r==4:expected[2:7]=[15,0,15,4,15]
    elif r==1 and p['ordinal']==0 and p['matrix_kind'] in (2,3):expected[2:6]=[1,0,1,1]
   elif name=='credit_status':
    expected=[0]*16
    if r==4:
     require(a[5]<=4,'Head receipt queue peak');expected[2:7]=[4,4,4,a[5],15]
    elif r==6:
     require(0<a[5]<=4 and a[9]<=96,'Origin grant window peak and ready count');expected[2:10]=[96,96,0,a[5],0,(1<<24)-1,96,a[9]]
    elif r==1 and p['ordinal']==0 and p['matrix_kind']==1:expected=[0,0,1,1,1,1,1]+[0]*9
   if expected is not None:require(a==expected,'Complete original lifecycle: '+name+' '+str(xy))
 def final_metadata(self,values,serial):
  position=1 if serial==2 else 0;generation=2 if serial==3 else 1
  for head,point in enumerate(self.plan['head_observers']):
   p=tuple(point);h=decode_head([int(v) for v in values['observer',p]],head,serial,position)
   require(h['coherent'] and h['field_bounds_valid'] and h['event']==11 and h['Q_and_native_retained'] and h['QK_error']==0 and h['attention_error']==0 and not h['protocol_error_latched'] and not h['sink_protocol_error'] and not h['source_cap_reached'] and h['archive_records_preceding_snapshot']==serial,'Complete physical head observer '+str(head))
  status=[[int(v) for v in values['qk_archive_status',tuple(p)]] for p in self.plan['head_observers']]
  archive=[[int(v) for v in values['qk_archive',tuple(p)]] for p in self.plan['head_observers']]
  decoded=[decode_archive(status[h],archive[h],head=h,identity=[1,generation,position,0,0],generation=serial) for h in range(24)]
  require(len(decoded)==24,'Every actual retained QK archive')
  return dict(heads=24,native_sources=16,archives=24,passed=True)
