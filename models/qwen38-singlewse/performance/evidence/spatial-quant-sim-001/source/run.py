"""Complete on-device spatial producer and original-weight consumer qualification."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 plan=json.loads(Path('region.json').read_text())['plan'];w,h=plan['width'],plan['height'];items=128//(w*h);local_words=items//2
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 chosen=list(range(meta['quant_groups'])) if a.physical else meta['simulation_groups'];rows=[]
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['input','codes','scale','packet','audit','ticks','weights','weight_scale','product','prepared']}
  opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,v,bits=32,root=False):
   ww,hh=(1,1) if root else (w,h);v=np.ascontiguousarray(v)
   runner.memcpy_h2d(ids[name],v.reshape(-1),0,0,ww,hh,v.size//(ww*hh),data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32,root=False):
   ww,hh=(1,1) if root else (w,h);v=np.zeros(ww*hh*count,np.uint32)
   runner.memcpy_d2h(v,ids[name],0,0,ww,hh,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(hh,ww,count)
  def ts(row,offset):return sum(int(row[offset+k])<<(16*k) for k in range(3))
  upload('weights',data['weights'].astype(np.uint32),16,True);upload('weight_scale',data['weight_scale'],root=True)
  started=time.perf_counter();runner.launch('prepare_weight',nonblock=False);np.testing.assert_array_equal(read('prepared',1,root=True),[[[1]]]);init_wall=time.perf_counter()-started
  t=read('ticks',18,16,True)[0,0];predecode=(ts(t,15)-ts(t,12))%(1<<48)
  expected=np.zeros((h,w,10),np.uint32);expected[:,:,[0,1,4,9]]=1
  for y in range(h):
   for x in range(w):
    children=int(x+1<w)+int(x==0 and y+1<h);expected[y,x,2]=children;expected[y,x,6]=children
    if x or y:expected[y,x,3]=1;expected[y,x,7]=1
  expected[0,0,5]=1;expected[0,0,8]=1
  for group in chosen:
   values=data['groups'][group].reshape(h,w,items);upload('input',values)
   t0=time.perf_counter();runner.launch('arm',nonblock=False);armed=read('audit',10)
   wanted_arm=np.zeros_like(armed);wanted_arm[:,:,0]=1;np.testing.assert_array_equal(armed,wanted_arm);arm_wall=time.perf_counter()-t0
   t0=time.perf_counter();runner.launch('start',nonblock=False)
   packet=read('packet',65);product=read('product',2,root=True).reshape(-1).view(np.float32);audit=read('audit',10);host=time.perf_counter()-t0
   np.testing.assert_array_equal(audit,expected)
   actual_codes=read('codes',items,16);q=data['quant'][group].astype(np.uint32).reshape(h,w,items);np.testing.assert_array_equal(actual_codes,q)
   s=int(data['scales'][group,0].view(np.uint32));np.testing.assert_array_equal(read('scale',1),np.full((h,w,1),s,np.uint32))
   half=(((q.reshape(-1)&127)<<7)|((q.reshape(-1)&128)<<8)).astype('<u2');packed=half.view('<u4')
   for y in range(h):
    for x in range(w):
     ranks=list(range(y*w,h*w)) if x==0 else list(range(y*w+x,(y+1)*w))
     words=np.concatenate([packed[r*local_words:(r+1)*local_words] for r in ranks])
     wanted=np.full(65,2774181210,np.uint32);wanted[:len(words)]=words;wanted[64]=s
     np.testing.assert_array_equal(packet[y,x],wanted)
   np.testing.assert_array_equal(read('input',items),values.view(np.uint32))
   np.testing.assert_array_equal(np.where(product==0,np.float32(0),product).view(np.uint32),np.where(data['ordered'][group]==0,np.float32(0),data['ordered'][group]).view(np.uint32))
   assert np.isfinite(product).all() and np.all(np.abs(product.astype(np.float64)-data['exact'][group])<=data['bounds'][group])
   t=read('ticks',18,16,True)[0,0];r=dict(group=group,packet_present_cycles=(ts(t,3)-ts(t,0))%(1<<48),producer_ready_cycles=(ts(t,6)-ts(t,0))%(1<<48),native_consumer_cycles=(ts(t,9)-ts(t,6))%(1<<48),producer_consumer_cycles=(ts(t,9)-ts(t,0))%(1<<48),host_arm_and_ready_seconds=arm_wall,host_start_through_result_audit_seconds=host,max_abs_fp64_error=float(np.max(np.abs(product.astype(np.float64)-data['exact'][group]))))
   rows.append(r);print(json.dumps(r),flush=True)
  np.testing.assert_array_equal(read('weights',128,16,True).reshape(-1),data['weights'])
  np.testing.assert_array_equal(read('weight_scale',1,root=True).reshape(-1),data['weight_scale'].view(np.uint32))
 r=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope=meta['scope'],application=[w,h],items_per_pe=items,fixture_sha256=meta['fixture_sha256'],groups=rows,clock_hz=None,
  weight_predecode_cycles=predecode,host_weight_prepare_seconds=init_wall,all_inputs_and_weight_retained=True,
  timing_note='Same-root maximum through scale, encoding, full ordered packet and native consumer. Input transfer and explicit receive arming/readiness audit are excluded from device interval and host setup is separate; start-entry skew remains. One constant original weight tile is predecoded once before groups; no host operation between quantization and consumer.')
 Path('result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,groups=len(rows))),flush=True)
if __name__=='__main__':main()
