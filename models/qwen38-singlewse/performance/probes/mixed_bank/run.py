"""Single upload, repeated FP8/BF16 resident dispatch, every subtree and full retention."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 w,h=meta['application'];owners=meta['owners'];chosen=meta['physical_cases'] if a.physical else meta['simulation_cases'];rows=[];replay={}
 document=json.loads(Path('region.json').read_text())
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','scale','bfweights','control','packet','local_result','aggregate','audit','ticks']};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,v,bits=32):
   v=np.ascontiguousarray(v);runner.memcpy_h2d(ids[name],v.reshape(-1),0,0,w,h,v.shape[-1],data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32):
   v=np.zeros(h*w*count,np.uint32);runner.memcpy_d2h(v,ids[name],0,0,w,h,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(h,w,count)
  def canon(v):return np.where(v==0,np.float32(0),v).astype(np.float32).view(np.uint32)
  print(json.dumps(dict(phase='initialize_all_resident_banks')),flush=True);started=time.perf_counter()
  upload('weights',data['weights'].astype(np.uint32),16);upload('scale',data['scales']);upload('bfweights',data['bfweights'].astype(np.uint32),16);initialization=time.perf_counter()-started
  for case in chosen:
   print(json.dumps(dict(phase='epoch',case=case)),flush=True);started=time.perf_counter();upload('control',data['control'][case].astype(np.uint32),16);upload('packet',data['packet'][case]);operand_upload=time.perf_counter()-started
   started=time.perf_counter();runner.launch('arm',nonblock=False);armed=read('audit',6);wanted=np.zeros_like(armed);wanted[:,:,0]=1;np.testing.assert_array_equal(armed,wanted);arm=time.perf_counter()-started
   started=time.perf_counter();runner.launch('start',nonblock=False);aggregate=read('aggregate',3);audit=read('audit',6);host=time.perf_counter()-started
   expected=np.zeros_like(audit);expected[:,:,[0,1,3,5]]=1
   for rank,(x,y) in enumerate(owners):expected[y,x,2]=len(document['trees'][0][rank]['children']);expected[y,x,4]=int(rank>0)
   np.testing.assert_array_equal(audit,expected)
   actual=np.ascontiguousarray(aggregate[:,:,:2]).view(np.float32);np.testing.assert_array_equal(canon(actual),canon(data['tree'][case]));np.testing.assert_array_equal(aggregate[:,:,2],data['count'][case])
   error=np.abs(actual.astype(np.float64)-data['exact'][case]);assert np.isfinite(actual).all() and np.all(error<=data['bounds'][case])
   np.testing.assert_array_equal(canon(read('local_result',2).view(np.float32)),canon(data['local'][case]));np.testing.assert_array_equal(read('packet',65),data['packet'][case]);np.testing.assert_array_equal(read('control',2,16),data['control'][case])
   if case in (0,1):replay[case]=aggregate.copy()
   if case in (8,9):np.testing.assert_array_equal(aggregate,replay[1 if case==8 else 0])
   ticks=read('ticks',6,16);x,y=owners[0];t=ticks[y,x];first=sum(int(t[k])<<(16*k) for k in range(3));last=sum(int(t[3+k])<<(16*k) for k in range(3))
   record=dict(**meta['cases'][case],root_contraction_cycles=(last-first)%(1<<48),max_abs_fp64_error=float(error.max()),host_operand_upload_seconds=operand_upload,host_arm_and_ready_seconds=arm,host_start_through_result_audit_seconds=host);rows.append(record);print(json.dumps(record),flush=True)
  print(json.dumps(dict(phase='full_bank_retention')),flush=True)
  np.testing.assert_array_equal(read('weights',112*128,16),data['weights']);np.testing.assert_array_equal(read('scale',112),data['scales'].view(np.uint32));np.testing.assert_array_equal(read('bfweights',12*256,16),data['bfweights'])
 r=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope=meta['scope'],application=[w,h],fixture_sha256=meta['fixture_sha256'],cases=rows,host_initialization_seconds=initialization,all_weights_retained=True,weight_uploads=1,clock_hz=None,
  timing_note='Root start to result includes typed slot dispatch, FP8 weight decode or BF16 streaming expansion, native dot, reduction and start-entry skew. Control/operand upload, receive arming/readiness and initialization are separate. Root timestamp can precede sender callbacks; all callbacks audited before rearm. No full-K/matrix/model rate.')
 Path('result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,epochs=len(rows))),flush=True)
if __name__=='__main__':main()
