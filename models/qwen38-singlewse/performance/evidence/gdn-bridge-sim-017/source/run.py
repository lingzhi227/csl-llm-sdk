"""Two exact cohost bodies, two ordered return domains, four continuing calls."""
import hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

plan=json.loads(Path('probe-plan.json').read_text());captures={};observed=[]
def phase(label,**values):print(json.dumps(dict(phase=label,**values)),flush=True)
with runtime(False)as(r,dtype,order):
 ids={name:r.get_id(name)for name in ('wire','received','probe_audit','bridge_status','bank')}
 opts=dict(data_type=dtype.MEMCPY_32BIT,streaming=False,order=order.ROW_MAJOR,nonblock=False)
 def read(name,x,n):
  a=np.zeros(n,np.uint32);r.memcpy_d2h(a,ids[name],x,0,1,1,n,**opts);return a
 banks={}
 for child in plan['children']:
  x=child['bridge_x'];banks[x]=np.arange(child['bank_words'],dtype=np.uint32)^np.uint32(0x9e3779b9+x)
  np.testing.assert_array_equal(read('bridge_status',x,8),np.zeros(8,np.uint32))
  r.memcpy_h2d(ids['bank'],banks[x],x,0,1,1,banks[x].size,**opts)
 for token in (1,2,99,100):
  wire=(np.arange(1161,dtype=np.uint32)*np.uint32(65537))^np.uint32(token*104729);wire[[0,387,774]]=token
  r.memcpy_h2d(ids['wire'],wire,0,0,1,1,1161,**opts)
  phase('arm',token=token);r.launch('bridge_stream_arm',np.uint32(token),nonblock=False)
  phase('send',token=token);r.launch('probe_send',nonblock=False)
  phase('wait',token=token);r.launch('probe_wait',nonblock=False)
  phase('returned',token=token)
  children=[]
  for child in plan['children']:
   x=child['bridge_x'];n=child['frames'];w=child['workers'];sink_x=child['sink_x']
   for _ in range(10):
    r.launch('bridge_inspect',nonblock=False);status=read('bridge_status',x,8)
    if status[0]==0:break
   np.testing.assert_array_equal(status,[0,0,0,9,n,w,token,1])
   sink=read('probe_audit',sink_x,8);np.testing.assert_array_equal(sink[:4],[1,n,3,w])
   np.testing.assert_array_equal(read('wire',sink_x,1161),wire)
   np.testing.assert_array_equal(read('bank',x,banks[x].size),banks[x])
   captures[f'bridge_{x}_{token}']=status
   captures[f'wire_{x}_{token}']=read('wire',sink_x,1161)
   children.append(dict(bridge_x=x,status=status.tolist(),sink=sink.tolist()))
  n=plan['frames'];actual=read('received',0,n*5).reshape(n,5)
  expected=np.array([[token,384*plan['group']+2*i,2,(token*104729)^i^0x3f807e11,5]for i in range(n)],np.uint32)
  np.testing.assert_array_equal(actual,expected)
  source=read('probe_audit',0,8);np.testing.assert_array_equal(source[:4],[1,n*5,plan['workers'],2]);assert source[4]>0
  captures[f'received_{token}']=actual
  observed.append(dict(token=token,source=source.tolist(),children=children))
 np.savez('actual.npz',**captures)
result=dict(passed=True,normal_stop=True,physical=False,neural_execution=False,full_model=False,synthetic_wire=True,
 invocations=observed,exact_forward_words=4*2*1161,exact_return_words=4*plan['frames']*5,
 raw_markers_received=4*plan['workers'],weighted_markers_emitted=8,switch_parent_executed=True,
 original_bank_extents_retained=True,bank_values_are_sentinels=True,
 capture_sha256=hashlib.sha256(Path('actual.npz').read_bytes()).hexdigest())
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');phase('passed')
