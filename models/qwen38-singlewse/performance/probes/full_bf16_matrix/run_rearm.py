"""Two-epoch control diagnosis on zero-initialized banks, not model qualification."""
import json,time
from pathlib import Path
import numpy as np
from backend import runtime

plan=json.loads(Path('region.json').read_text());assert plan['diagnostic_simprint']
w,h=plan['application'];workers=plan['workers'];assert [w,h]==[631,2]
started=time.monotonic();records=[]
def mark(phase,**extra):print(json.dumps(dict(phase=phase,wall_seconds=time.monotonic()-started,**extra)),flush=True)
expected=np.zeros((h,w,12),np.uint32);expected[...,0]=1;expected[...,10]=1
for x,y in plan['input_path']:expected[y,x,8]=1
expected[0,0,[1,7,9]]=1
children={};pending=[(0,40)]
while pending:
 first,size=pending.pop();left=size//2;right=size-1-left;children[first]=(bool(left),bool(right))
 if left:pending.append((first+1,left))
 if right:pending.append((first+1+left,right))
for node in workers:
 x,y=node['xy'];k=node['k'];expected[y,x,[1,2,5,7]]=1;expected[y,x,3:5]=children[k]
 if k==0:
  g=node['group'];steps=0
  while g%(2**(steps+1))==0 and g+2**steps<24:steps+=1
  pairs=min(2**steps,24-g);expected[y,x,6]=pairs-1;expected[y,x,7]=pairs
with runtime(False) as (runner,dtype,order):
 ids={n:runner.get_id(n) for n in ['control','packet','audit','gather']}
 opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
 def upload(name,data,x,y,width,height,bits):
  data=np.ascontiguousarray(data,dtype=np.uint32);runner.memcpy_h2d(ids[name],data.reshape(-1),x,y,width,height,data.size//(width*height),data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
 def read(name,n,width=w,height=h,bits=16):
  data=np.zeros(width*height*n,np.uint32);runner.memcpy_d2h(data,ids[name],0,0,width,height,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return data.reshape(height,width,n)
 np.testing.assert_array_equal(read('audit',1),0);mark('zero_bank_initialization_observed')
 for epoch in range(2):
  control=np.zeros((h,w,2),np.uint32)
  for node in workers:control[node['xy'][1],node['xy'][0]]=[1,6]
  stream=np.full((40,65),0x3f803f80,np.uint32);stream[:,64]=np.arange(40,dtype=np.uint32)+epoch*40
  upload('control',control,0,0,w,h,16);upload('packet',stream,0,0,1,1,32)
  mark('arm_submitted',epoch=epoch);runner.launch('arm',nonblock=False)
  np.testing.assert_array_equal(read('audit',1),1);mark('all_pes_armed',epoch=epoch)
  runner.launch('start',nonblock=True);mark('start_submitted',epoch=epoch)
  audit=read('audit',12);np.testing.assert_array_equal(audit,expected);mark('complete_counters_observed',epoch=epoch)
  values=read('gather',48,1,1,32).view(np.float32);np.testing.assert_array_equal(values,0)
  records.append(dict(epoch=epoch,counters_exact=True,zero_matrix_outputs_exact=True));mark('zero_outputs_observed',epoch=epoch)
result=dict(passed=True,normal_stop=True,physical=False,diagnostic=True,full_model=False,application=[w,h],epochs=records,
            original_target_weights_retained=False,all_loaded_weights_retained=False,
            scope='Synthetic zero-initialized BF16 banks: two full631x2 input/tree/gather/rearm epochs with complete callback/teardown and zero-output checks. No original weight upload, retention or numerical qualification.')
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');mark('diagnostic_passed')
