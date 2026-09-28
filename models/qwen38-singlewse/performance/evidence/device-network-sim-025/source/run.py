"""Actual addressed/tagged fabric control through a shared credited return bus."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime
from device_client import Encoder,Client,batches

meta=json.loads(Path('fixture.json').read_text());config=json.loads(Path('selection.json').read_text())
assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
counts={p['tag']:0 for p in config['workers']};total=0;ping_count=0;last=None;started=time.monotonic()
gx,gy=config['gateway'];order_indices=[0,7,1,6,2,5,3,4]
print(json.dumps(dict(phase='runtime_enter')),flush=True)
with runtime(False) as (runner,dtype,order):
 print(json.dumps(dict(phase='runtime_ready')),flush=True)
 ids={n:runner.get_id(n) for n in ['device_command','device_response','device_audit','sentinel','operand40','operand48','operand_counts']}
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
  runner.memcpy_h2d(14,submit,gx,gy,1,1,4,**dict(opts,streaming=True))
  deadline=time.monotonic()+10
  while True:
   status=read('device_audit',4)
   if status[0]==total:break
   assert status[0]==total-1 and time.monotonic()<deadline,('device completion',status.tolist(),last)
  np.testing.assert_array_equal(status,last)
  response=read('device_response',4+reply_words)
  np.testing.assert_array_equal(response[:4],[sequence,0,count,bits if count else 32])
  if total%32==0:np.testing.assert_array_equal(read('device_audit',4),last);ping()
  return response[4:].view(np.uint16) if bits==16 else response[4:]
 ping();np.testing.assert_array_equal(read('device_audit',4),0)
 # Qualify widths, source switches and zero-work reuse before spending the
 # bounded simulator budget on every original bank word.
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
 print(json.dumps(dict(phase='protocol_checks_passed',commands=total,elapsed_seconds=time.monotonic()-started)),flush=True)
 encoder=Encoder(config['internal_target_count']);encoder.total=total;encoder.sequences=dict(counts)
 client=Client(runner,dtype,order,(gx,gy),progress=lambda p:print(json.dumps(dict(phase='inline_progress',**p)),flush=True));client.completed=total
 observed=[0,0]
 for case in meta['operand_cases']:
  n=case['case'];frame=encoder.operand(case['invocation'],case['group'],case['parts'],data['operand_'+str(n)])
  client.stream([frame]);slot=0 if case['parts']==40 else 1;observed[slot]+=1
  deadline=time.monotonic()+10
  while True:
   actual=read('operand_counts',2,3,2)
   if np.array_equal(actual,observed):break
   assert time.monotonic()<deadline,('operand consumer',actual.tolist(),observed)
  np.testing.assert_array_equal(read('operand'+str(case['parts']),133,3,2),data['packet_'+str(n)])
  last=list(frame.audit)
 print(json.dumps(dict(phase='arrival_operand_checks_passed',cases=len(meta['operand_cases']),elapsed_seconds=time.monotonic()-started)),flush=True)
 batch=[encoder.command(config['workers'][i]['tag'],0,0,len(data['bank_'+str(i)])-8,8,payload=changed[i]) for i in (0,7)]
 client.stream(batch)
 for i in (7,0):
  frame=encoder.command(config['workers'][i]['tag'],1,0,len(data['bank_'+str(i)])-8,8);client.stream([frame])
  np.testing.assert_array_equal(client.read_reply(),changed[i])
 print(json.dumps(dict(phase='two_command_batch_passed',commands=client.completed)),flush=True)
 # Interleave different targets for every256-word stripe to exercise old-source
 # transit restoration before traffic from the next upstream/downstream source.
 def writes():
  for first in range(0,max(len(data['bank_'+str(i)]) for i in order_indices),256):
   for i in order_indices:
    block=data['bank_'+str(i)][first:first+256]
    if len(block):yield encoder.command(config['workers'][i]['tag'],0,0,first,len(block),payload=block)
 bulk_streams=0;bulk_commands=0
 for batch in batches(writes(),16384):
  client.stream(batch);bulk_streams+=1;bulk_commands+=len(batch);last=list(batch[-1].audit);ping()
  print(json.dumps(dict(phase='inline_bank_batch',commands=client.completed,batch_commands=len(batch),batch_words=sum(len(f.wire) for f in batch))),flush=True)
 print(json.dumps(dict(phase='all_original_banks_loaded',commands=client.completed,elapsed_seconds=time.monotonic()-started)),flush=True)
 for first in range(0,max(len(data['bank_'+str(i)]) for i in order_indices),256):
  for i in reversed(order_indices):
   block=data['bank_'+str(i)][first:first+256]
   if len(block):
    frame=encoder.command(config['workers'][i]['tag'],1,0,first,len(block));client.stream([frame]);last=list(frame.audit)
    np.testing.assert_array_equal(client.read_reply(),block)
    if client.completed%32==0:ping()
 total=client.completed;counts=encoder.sequences
 np.testing.assert_array_equal(read('device_audit',4),last);ping()
 result=dict(passed=True,physical=False,normal_stop=True,commands=total,commands_by_tag=counts,sdk_east_sentinel_rounds=ping_count,
  sdk_east_sentinel_pes=3,selected_complete_original_banks=len(config['workers']),original_selected_bank_words=meta['total_selected_bank_words'],
  all_selected_banks_loaded_and_retained=True,all_descriptor16_values_exact=True,high_half32_values_exact=True,
  gateway_only_worker_access=True,interleaved_return_sources=True,standby_zero_work_epochs=3,
  legacy_protocol_commands=94,inline_bulk_commands=bulk_commands,inline_bulk_sdk_stream_calls=bulk_streams,
  maximum_inline_batch_words=16384,arrival_operand_cases=len(meta['operand_cases']),arrival_operand_packets_exact=True,
  operand_preparation_executed=True,complete_contraction_executed=False,
  neural_execution=False,complete_stage=False,full_model_speed_target_achieved=False,
  simulator_host_seconds=time.monotonic()-started)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
