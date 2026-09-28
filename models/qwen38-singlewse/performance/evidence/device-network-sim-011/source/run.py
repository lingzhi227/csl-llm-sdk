"""Actual addressed/tagged fabric control through a shared credited return bus."""
import hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

meta=json.loads(Path('fixture.json').read_text());config=json.loads(Path('selection.json').read_text())
assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
counts={p['tag']:0 for p in config['workers']};total=0;ping_count=0;last=None
gx,gy=config['gateway'];order_indices=[0,7,1,6,2,5,3,4]
print(json.dumps(dict(phase='runtime_enter')),flush=True)
with runtime(False) as (runner,dtype,order):
 print(json.dumps(dict(phase='runtime_ready')),flush=True)
 ids={n:runner.get_id(n) for n in ['device_command','device_response','device_audit','device_submit','sentinel']}
 opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
 def read(name,n,x=gx,y=gy,height=1):
  result=np.zeros(n*height,np.uint32);runner.memcpy_d2h(result,ids[name],x,y,1,height,n,**opts);return result
 def ping():
  global ping_count
  ping_count+=1;value=0xf1230000+ping_count
  print(json.dumps(dict(phase='sentinel_ping',round=ping_count)),flush=True)
  sentinels=np.tile(np.asarray([value,ping_count],np.uint32),3)
  runner.memcpy_h2d(ids['sentinel'],sentinels,3,0,1,3,2,**opts)
  print(json.dumps(dict(phase='sentinel_read',round=ping_count)),flush=True)
  np.testing.assert_array_equal(read('sentinel',2,3,0,3).reshape(3,2),np.tile([value,ping_count],(3,1)))
 def command(i,op,a=0,b=0,c=0,d=0,payload=None,bits=32):
  global total,last
  target=config['workers'][i]['tag'];counts[target]+=1;sequence=counts[target];total+=1;last=[total,target,sequence,op]
  wire=np.asarray([sequence,op,a,b,c,d]+([] if payload is None else list(map(int,payload))),np.uint32)
  count=c if op==1 else 0
  reply_words=count//2 if bits==16 else count
  if bits==16:assert op==1 and a==1 and b%2==0 and count%2==0
  print(json.dumps(dict(phase='exchange',command=total,target=target,sequence=sequence,opcode=op)),flush=True)
  runner.memcpy_h2d(ids['device_command'],wire,gx,gy,1,1,len(wire),**opts)
  submit=np.asarray([target,len(wire),4+reply_words,total],np.uint32)
  runner.memcpy_h2d(ids['device_submit'],submit,gx,gy,1,1,4,**opts)
  response=read('device_response',4+reply_words)
  np.testing.assert_array_equal(response[:4],[sequence,0,count,bits if count else 32])
  if total%32==0:np.testing.assert_array_equal(read('device_audit',4),last);ping()
  return response[4:].view(np.uint16) if bits==16 else response[4:]
 ping();np.testing.assert_array_equal(read('device_audit',4),0)
 # Interleave different targets for every256-word stripe to exercise old-source
 # transit restoration before traffic from the next upstream/downstream source.
 for first in range(0,max(len(data['bank_'+str(i)]) for i in order_indices),256):
  for i in order_indices:
   block=data['bank_'+str(i)][first:first+256]
   if len(block):command(i,0,0,first,len(block),payload=block)
 for i in order_indices:
  command(i,0,1,0,40,payload=data['setup_'+str(i)])
  np.testing.assert_array_equal(command(i,1,1,0,40,bits=16),data['setup_'+str(i)])
  half=np.asarray([0xffff,0x8000,0,1,0x7fff,0xfedc],np.uint16)
  command(i,0,1,2,6,payload=half)
  np.testing.assert_array_equal(command(i,1,1,2,6,bits=16),half)
  command(i,0,1,2,6,payload=data['setup_'+str(i)][2:8])
 changed={}
 for i in order_indices:
  changed[i]=np.asarray([0xffffffff,0,0x12345678,0x80000001,0x0001ffff,0xffff0001,0x7f800001,(config['workers'][i]['tag']+1)*2654435761 & 0xffffffff],np.uint32)
  command(i,0,0,len(data['bank_'+str(i)])-8,8,payload=changed[i])
 for i in reversed(order_indices):np.testing.assert_array_equal(command(i,1,0,len(data['bank_'+str(i)])-8,8),changed[i])
 for i in order_indices:command(i,0,0,len(data['bank_'+str(i)])-8,8,payload=data['bank_'+str(i)][-8:])
 for epoch in (1,2,3):
  for i in (0,1):
   assert data['setup_'+str(i)][17]==0
   command(i,2,epoch);command(i,3);command(i,4);command(i,5,epoch,epoch,2)
   np.testing.assert_array_equal(command(i,1,2,0,6),[epoch,0,0,0,0,1])
 for first in range(0,max(len(data['bank_'+str(i)]) for i in order_indices),256):
  for i in reversed(order_indices):
   block=data['bank_'+str(i)][first:first+256]
   if len(block):np.testing.assert_array_equal(command(i,1,0,first,len(block)),block)
 np.testing.assert_array_equal(read('device_audit',4),last);ping()
 result=dict(passed=True,physical=False,normal_stop=True,commands=total,commands_by_tag=counts,sdk_east_sentinel_rounds=ping_count,
  sdk_east_sentinel_pes=3,selected_complete_original_banks=len(config['workers']),original_selected_bank_words=meta['total_selected_bank_words'],
  all_selected_banks_loaded_and_retained=True,all_descriptor16_values_exact=True,high_half32_values_exact=True,
  gateway_only_worker_access=True,interleaved_return_sources=True,standby_zero_work_epochs=3,
  neural_execution=False,complete_stage=False,full_model_speed_target_achieved=False)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
