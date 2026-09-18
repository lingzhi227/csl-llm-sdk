"""Offline core evidence for synthetic communication; incomplete is not deadlock."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np

ROOT=Path.cwd();START=time.monotonic()
assert sorted(os.sched_getaffinity(0))==[0]
plan=json.loads((ROOT/'plan.json').read_bytes())
expected_meta=np.asarray(plan['metadata'],np.uint16)
names=json.loads((ROOT/'offline-symbol-names.json').read_bytes())
inventory=json.loads((ROOT/'offline-inventory.json').read_bytes())
programs={}
for program in inventory['programs']:
 for x,y,w,h in program['rectangles']:
  for xx in range(x,x+w):
   for yy in range(y,y+h):
    assert (xx,yy) not in programs
    programs[xx,yy]=program['symbols']
memories={(r['x'],r['y']):r for r in json.loads((ROOT/'csdb-memory-words.json').read_bytes())}
class CoreWords:
 def get_symbol_rect(self,rect,name,dtype):
  (col,row),(width,height)=rect;result=[]
  for x in range(col,col+width):
   column=[]
   for y in range(row,row+height):
    spec=programs[x,y][name];assert spec['byte_address']%2==spec['bytes']%2==0
    record=memories[x,y];offset=spec['byte_address']//2-record['word_address'];length=spec['bytes']//2
    assert offset>=0 and offset+length<=len(record['words'])
    value=np.asarray(record['words'][offset:offset+length],dtype='<u2').view(dtype)
    column.append(value)
   result.append(column)
  return np.asarray(result,dtype=dtype)
debug=CoreWords()
arrays={};reads=[]
def read(key,rect,dtype,symbol=None):
 name=names[key] if symbol is None else symbol
 value=debug.get_symbol_rect(rect,name,np.dtype(dtype))
 arrays[key]=value
 reads.append(dict(key=key,symbol=name,rectangle=rect,dtype=str(value.dtype),shape=list(value.shape),bytes=value.nbytes))
 return value
state=read('state',((4,1),(37,10)),np.uint32).transpose(1,0,2)
metadata=read('metadata',((4,1),(37,10)),np.uint16).transpose(1,0,2)
assert state.shape==(10,37,16) and np.array_equal(metadata,expected_meta)
assert np.array_equal(state[:,:,1],metadata[:,:,0])
data=read('data',((5,1),(24,1)),np.uint16)[:,0,:]
q=read('query_counts',((5,1),(24,1)),np.uint16)[:,0,:]
k=read('key_counts',((5,1),(24,1)),np.uint16)[:,0,:]
v=read('value_counts',((5,1),(24,1)),np.uint16)[:,0,:]
assert data.shape==(24,1024) and q.shape==(24,4) and k.shape==v.shape==(24,2)
assert max(q.max(),k.max(),v.max())<=64
head_flags={}
for field in ['packets.sending','packets.receiving','ack_pending','entered','waiting','done']:
 head_flags[field]=read('head_'+field,((5,1),(24,1)),np.uint16,field)[:,0,0]
root_flags={}
for block,rect in enumerate([((5,2),(1,9)),((29,2),(12,8)),((29,10),(7,1))]):
 (col,row),(width,height)=rect
 for field in ['packets.sending','rows.active','rows.sending','rows.offset','fanout_index','rows.destination_x','root_pending','waiting','done']:
  raw=read('root_'+str(block)+'_'+field,rect,np.uint16,field)
  for xx in range(width):
   for yy in range(height):root_flags.setdefault((col+xx-4,row+yy-1),{})[field]=int(raw[xx,yy,0])
heads=[];committed_bad=0
for head in range(24):
 expected=np.asarray([0x1000+(4*head+i//128)*128+i%128 for i in range(512)]+[0x4000+(2*(head//6)+i//128)*128+i%128 for i in range(256)]+[0x4400+(2*(head//6)+i//128)*128+i%128 for i in range(256)],np.uint16)
 counts=np.concatenate([q[head],k[head],v[head]])
 checked=0;bad=0
 for group,count in enumerate(counts):
  length=2*int(count);start=group*128
  checked+=length;bad+=int(np.count_nonzero(data[head,start:start+length]!=expected[start:start+length]))
 committed_bad+=bad
 s=state[0,head+1]
 heads.append(dict(head=head,state=s.tolist(),flags={field:int(raw[head]) for field,raw in head_flags.items()},query_counts=q[head].tolist(),key_counts=k[head].tolist(),value_counts=v[head].tolist(),committed_channels=checked,committed_mismatches=bad,all_payload_mismatches=int(np.count_nonzero(data[head]!=expected)),complete=bool(s[0]==3 and s[2]==24 and s[3]==1 and s[5]==7 and np.all(counts==64) and np.array_equal(data[head],expected))))
roots=[]
for y in range(10):
 for x in range(37):
  m=metadata[y,x];s=state[y,x]
  if m[0]!=1:continue
  fanout=int(m[3]);roots.append(dict(application=[x,y],physical=[x+4,y+1],kind=int(m[1]),group=int(m[2]),fanout=fanout,destination_start=int(m[4]),flags=root_flags[(x,y)],state=s.tolist(),complete=bool(s[0]==3 and s[2]==0 and s[3]==fanout*3 and s[4]==fanout)))
assert len(roots)==112
origin=state[0,0]
complete=bool(np.all(state[:,:,0]==3) and committed_bad==0 and all(h['complete'] for h in heads) and all(r['complete'] for r in roots) and origin[2]==24 and origin[6]==0xffffff)
np.savez(ROOT/'csdb-arrays.npz',**arrays)
report=dict(status='complete' if complete else 'partial_snapshot',completed_graph=complete,deadlock_claimed=False,hardware=False,neural_math=False,origin=origin.tolist(),heads=heads,roots=roots,state_counts={str(int(x)):int(np.count_nonzero(state[:,:,0]==x)) for x in np.unique(state[:,:,0])},completed_heads=[h['head'] for h in heads if h['complete']],head_row_packets_received=sum(h['state'][2] for h in heads),root_packets_sent=sum(r['state'][3] for r in roots),source_row_completions=sum(r['state'][4] for r in roots),committed_payload_mismatches=committed_bad,reads=reads,seconds=time.monotonic()-START)
raw=(json.dumps(report,indent=2)+'\n').encode();assert len(raw)<=262144
with (ROOT/'csdb-audit.json').open('xb') as f:f.write(raw)
print(json.dumps({key:report[key] for key in ['status','completed_graph','completed_heads','state_counts','head_row_packets_received','root_packets_sent','source_row_completions','committed_payload_mismatches','seconds']}))
