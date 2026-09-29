"""Synthetic wire/lifetime qualification on one exact original bank-extent cohost."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

plan=json.loads(Path('probe-plan.json').read_text());captures={};observed=[]
frames=plan['frames'];workers=plan['workers'];words=5*frames;row_start=384*plan['group']
def phase(label,**values):print(json.dumps(dict(phase=label,**values)),flush=True)
phase('runtime_create')
with runtime(False)as(r,dtype,order):
 phase('runtime_loaded')
 ids={name:r.get_id(name)for name in ('wire','received','probe_audit','bridge_status','bank')}
 opts=dict(data_type=dtype.MEMCPY_32BIT,streaming=False,order=order.ROW_MAJOR,nonblock=False)
 def read(name,x,n):
  phase('read',name=name,x=x,words=n)
  a=np.zeros(n,np.uint32);r.memcpy_d2h(a,ids[name],x,0,1,1,n,**opts);return a
 bank=np.arange(plan['bank_words'],dtype=np.uint32)^np.uint32(0x9e3779b9)
 phase('preflight_ids',ids={k:str(v)for k,v in ids.items()})
 np.testing.assert_array_equal(read('bridge_status',1,8),np.zeros(8,np.uint32))
 phase('preflight_status_passed')
 phase('bank_upload',words=bank.size)
 r.memcpy_h2d(ids['bank'],bank,1,0,1,1,bank.size,**opts)
 for token in (1,2,99,100):
  wire=(np.arange(1161,dtype=np.uint32)*np.uint32(65537))^np.uint32(token*104729)
  wire[[0,387,774]]=token
  phase('wire_upload',token=token)
  r.memcpy_h2d(ids['wire'],wire,0,0,1,1,1161,**opts)
  phase('arm',token=token);r.launch('probe_arm',nonblock=False)
  phase('send',token=token);r.launch('probe_send',nonblock=False)
  phase('wait',token=token);r.launch('probe_wait',nonblock=False)
  phase('wait_returned',token=token)
  phase('preinspect_source',value=read('probe_audit',0,8).tolist())
  phase('preinspect_sink',value=read('probe_audit',2,8).tolist())
  phase('preinspect_bank',value=read('bank',1,8).tolist())
  for _ in range(10):
   phase('inspect',token=token)
   r.launch('bridge_inspect',nonblock=False);status=read('bridge_status',1,8)
   if status[0]==0:break
  np.testing.assert_array_equal(status,[0,0,0,9,frames,workers,token,1])
  np.testing.assert_array_equal(read('wire',2,1161),wire)
  actual=read('received',0,words).reshape(frames,5)
  expected=np.empty((frames,5),np.uint32);expected[:,0]=token;expected[:,1]=row_start+2*np.arange(frames,dtype=np.uint32);expected[:,2]=2
  expected[:,3]=np.uint32(token*104729)^np.arange(frames,dtype=np.uint32)^np.uint32(0x3f807e11);expected[:,4]=5
  np.testing.assert_array_equal(actual,expected)
  source=read('probe_audit',0,8);sink=read('probe_audit',2,8)
  assert tuple(source[:4])==(1,words,workers,1)and source[4]>0
  assert tuple(sink[:4])==(1,frames,3,workers)
  np.testing.assert_array_equal(read('bank',1,bank.size),bank)
  captures['received_'+str(token)]=actual;captures['bridge_'+str(token)]=status
  observed.append(dict(token=token,source=source.tolist(),sink=sink.tolist(),bridge=status.tolist()))
 phase('capture');np.savez('actual.npz',**captures)
result=dict(passed=True,normal_stop=True,physical=False,neural_execution=False,full_model=False,synthetic_wire=True,
 original_cohost=plan['original_cohost'],original_bank_words=bank.size,original_bank_extent_retained=True,bank_values_are_sentinels=True,
 invocations=observed,exact_forward_words=4*1161,exact_return_words=4*words,raw_markers_received=4*workers,weighted_markers_emitted=4,
 early_reverse_transfer_in_every_invocation=True,local_callback_drained_every_invocation=True,
 capture_sha256=hashlib.sha256(Path('actual.npz').read_bytes()).hexdigest())
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,scope='Synthetic bidirectional transport, not neural inference')),flush=True)
