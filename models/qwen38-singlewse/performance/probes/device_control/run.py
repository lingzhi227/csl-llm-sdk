"""Host addresses the SDK gateway only; every worker access crosses the fabric."""
import hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
sequence=0;records=[]
def progress(phase,**fields):print(json.dumps(dict(phase=phase,sequence=sequence,**fields)),flush=True)
progress('runtime_enter')
with runtime(False) as (runner,dtype,order):
 progress('runtime_ready')
 ids={n:runner.get_id(n) for n in ['command','response','audit']}
 opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
 def read(name,n):
  progress('read_begin',name=name,words=n)
  out=np.zeros(n,np.uint32);runner.memcpy_d2h(out,ids[name],0,0,1,1,n,**opts);progress('read_done',name=name);return out
 def command(op,a=0,b=0,c=0,d=0,payload=None,bits=32):
  global sequence
  sequence+=1;header=[sequence,op,a,b,c,d];wire=np.asarray(header+([] if payload is None else list(map(int,payload))),np.uint32)
  count=c if op==1 else 0;progress('upload_begin',opcode=op);runner.memcpy_h2d(ids['command'],wire,0,0,1,1,len(wire),**opts)
  progress('launch_begin',opcode=op)
  runner.launch('exchange',np.uint16(len(wire)),np.uint16(4+count),nonblock=False)
  progress('launch_done',opcode=op)
  response=read('response',4+count);np.testing.assert_array_equal(response[:4],[sequence,0,count,bits if count else 32])
  np.testing.assert_array_equal(read('audit',4),[sequence,sequence,len(wire),4+count])
  return response[4:] & (65535 if bits==16 else np.uint32(0xffffffff))
 def store(segment,values):
  for first in range(0,len(values),256):
   block=values[first:first+256];command(0,segment,first,len(block),payload=block)
 def load(segment,n,bits=32):return np.concatenate([command(1,segment,first,min(256,n-first),bits=bits) for first in range(0,n,256)])
 np.testing.assert_array_equal(read('audit',4),0)
 store(0,data['bank']);store(1,data['setup'])
 np.testing.assert_array_equal(load(0,len(data['bank'])),data['bank'])
 np.testing.assert_array_equal(load(1,40,16),data['setup'])
 for epoch in [1,2,3]:
  command(2,epoch);command(3);command(4)
  np.testing.assert_array_equal(command(1,8,0,4),[epoch,0,0,0])
  # This selected original tail PE has no A tiles. It exercises real begin,
  # generation checks, drain callback and ack without inventing a neural output.
  assert data['setup'][17]==0
  command(5,epoch,epoch,2)
  status=command(1,3,0,4);assert status[2]==0 and status[3]==epoch
  np.testing.assert_array_equal(command(1,2,0,6),[epoch,0,0,0,0,1])
  records.append(dict(epoch=epoch,drained=True,no_neural_work=True))
 # A deliberately changed tail is fully checked and restored, then the whole
 # original bank is re-read across the same credited transport.
 changed=np.array([0x12345678,0xffffffff,0,0x7f800001],np.uint32);first=len(data['bank'])-4
 command(0,0,first,4,payload=changed);np.testing.assert_array_equal(command(1,0,first,4),changed)
 command(0,0,first,4,payload=data['bank'][first:]);np.testing.assert_array_equal(load(0,len(data['bank'])),data['bank'])
 result=dict(passed=True,physical=False,normal_stop=True,commands=sequence,original_bank_words=len(data['bank']),
  original_bank_loaded_and_retained=True,descriptor16_bit_exact=True,gateway_only_host_access=True,
  complete_neural_projection=False,full_model=False,full_model_speed_target_achieved=False,epochs=records)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
