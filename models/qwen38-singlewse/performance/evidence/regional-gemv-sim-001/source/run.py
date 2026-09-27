"""Matched serialized/overlapped regional GEMV, exact traffic and numerical audit."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 w,h=meta['application'];rows=[]
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','raw','packet','scale','output','row_result','ticks','audit']}
  opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,v,bits=32):
   v=np.ascontiguousarray(v);runner.memcpy_h2d(ids[name],v.reshape(-1),0,0,w,h,v.shape[-1],data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32,kind=np.float32):
   v=np.zeros(h*w*count,kind);runner.memcpy_d2h(v,ids[name],0,0,w,h,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(h,w,count)
  def bits(v):return np.where(v==0,np.float32(0),v).astype(np.float32).view(np.uint32)
  expected_audit=np.zeros((h,w,8),np.uint32);expected_audit[:,:,:4]=1;expected_audit[0,:,4]=1;expected_audit[:,0,5]=1;expected_audit[:-1,0,6]=1;expected_audit[1:,0,7]=1
  for case in range(meta['cases']):
   upload('weights',data['weights'][case].astype(np.uint32),16);upload('scale',data['scales'][case]);upload('raw',data['raw'][case])
   pair={}
   for mode in ([0,1] if case%2==0 else [1,0]):
    started=time.perf_counter();runner.launch('execute',np.uint16(mode),nonblock=False)
    actual=read('output',2);result=read('row_result',2);audit=read('audit',8,kind=np.uint32)
    host=time.perf_counter()-started
    np.testing.assert_array_equal(audit,expected_audit)
    assert np.isfinite(actual).all() and np.all(np.abs(actual.astype(np.float64)-data['exact'][case])<=data['bounds'][case])
    np.testing.assert_array_equal(bits(actual),bits(data['ordered'][case]))
    np.testing.assert_array_equal(bits(result[:,0]),bits(actual[:,-1]))
    np.testing.assert_array_equal(read('packet',65,kind=np.uint32),data['wire'][case])
    np.testing.assert_array_equal(read('weights',128,16,np.uint32),data['weights'][case])
    np.testing.assert_array_equal(read('raw',33,kind=np.uint32),data['raw'][case])
    np.testing.assert_array_equal(read('scale',1),data['scales'][case])
    ticks=read('ticks',24,16,np.uint32);timing=[]
    for y in range(h):
     for x in range(w):
      t=[sum(int(ticks[y,x,j+k])<<(16*k) for k in range(3)) for j in range(0,24,3)]
      timing.append(dict(x=x,y=y,entry_to_finish=(t[7]-t[0])%(1<<48),decode=(t[2]-t[1])%(1<<48),dot=(t[5]-t[4])%(1<<48),operand_ready_callback=(t[3]-t[0])%(1<<48),sum_send_done=(t[6]-t[0])%(1<<48)))
    pair[mode]=actual.copy();r=dict(case=case,overlapped=bool(mode),root_region_cycles=timing[0]['entry_to_finish'],host_launch_through_result_audit_seconds=host,timing=timing,max_abs_fp64_error=float(np.max(np.abs(actual.astype(np.float64)-data['exact'][case]))))
    rows.append(r);print(json.dumps({k:v for k,v in r.items() if k!='timing'}),flush=True)
   np.testing.assert_array_equal(bits(pair[0]),bits(pair[1]))
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope='Original projection submatrices: on-device exact operand coding, column multicast, local FP8 dot, ascending-K sums, returned rows and root completion; no full model',
  application=[w,h],fixture_sha256=meta['fixture_sha256'],cases=rows,clock_hz=None,all_inputs_retained=True,
  timing_note='Root same-PE entry-to-all-row completion includes host-launch entry skew and device encoding/decode/transport/dot/reduction. No cross-PE timestamp subtraction. Host time includes result/audit transfer; local readiness timestamps mark callbacks, not fabric arrival.')
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,epochs=len(rows))),flush=True)
if __name__=='__main__':main()
