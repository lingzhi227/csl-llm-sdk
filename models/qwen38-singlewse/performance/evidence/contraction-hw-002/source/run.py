"""Full-K tree/chain numerical qualification with separate root clocks and host setup."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 w,h=meta['application'];blocks=meta['blocks'];owners=meta['owners'];chosen=meta['physical_cases'] if a.physical else meta['simulation_cases'];rows=[]
 document=json.loads(Path('region.json').read_text())
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','scale','raw','packet','local_result','aggregate','audit','ticks']};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,v,bits=32):
   v=np.ascontiguousarray(v);runner.memcpy_h2d(ids[name],v.reshape(-1),0,0,w,h,v.shape[-1],data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32):
   v=np.zeros(h*w*count,np.uint32);runner.memcpy_d2h(v,ids[name],0,0,w,h,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(h,w,count)
  def canon(v):return np.where(v==0,np.float32(0),v).astype(np.float32).view(np.uint32)
  for case in chosen:
   print(json.dumps(dict(phase='upload',case=case)),flush=True)
   upload('weights',data['weights'][case].astype(np.uint32),16);upload('scale',data['scales'][case]);upload('raw',data['raw'][case])
   print(json.dumps(dict(phase='uploaded',case=case)),flush=True)
   for mode in ([0,1] if case%2==0 else [1,0]):
    kind='tree' if mode else 'chain';print(json.dumps(dict(phase='arm',case=case,schedule=kind)),flush=True);started=time.perf_counter();runner.launch('arm',np.uint16(mode),nonblock=False);armed=read('audit',6);wanted=np.zeros_like(armed);wanted[:,:,0]=1;np.testing.assert_array_equal(armed,wanted);arm=time.perf_counter()-started
    print(json.dumps(dict(phase='start',case=case,schedule=kind)),flush=True)
    started=time.perf_counter();runner.launch('start',nonblock=False);aggregate=read('aggregate',3);audit=read('audit',6);host=time.perf_counter()-started
    expected=np.zeros_like(audit);expected[:,:,[0,5]]=1
    for group,n in enumerate(blocks):
     for k,(x,y) in enumerate(owners[group]):
      expected[y,x,1]=1;expected[y,x,3]=1;expected[y,x,4]=int(k>0)
      expected[y,x,2]=len(document['trees'][group][k]['children']) if mode else int(k+1<n)
    np.testing.assert_array_equal(audit,expected);print(json.dumps(dict(phase='completion_audit_passed',case=case,schedule=kind)),flush=True)
    actual=np.ascontiguousarray(aggregate[:,:,:2]).view(np.float32);np.testing.assert_array_equal(canon(actual),canon(data[kind][case]));np.testing.assert_array_equal(aggregate[:,:,2],data[kind+'_count'][case])
    error=np.abs(actual.astype(np.float64)-data[kind+'_exact'][case]);assert np.isfinite(actual).all() and np.all(error<=data[kind+'_bounds'][case])
    np.testing.assert_array_equal(canon(read('local_result',2).view(np.float32)),canon(data['local'][case]))
    np.testing.assert_array_equal(read('packet',65),data['wire'][case])
    ticks=read('ticks',6,16);timing=[]
    for group,n in enumerate(blocks):
     x,y=owners[group][0];t=ticks[y,x];first=sum(int(t[k])<<(16*k) for k in range(3));last=sum(int(t[3+k])<<(16*k) for k in range(3));timing.append(dict(row=group,root=[x,y],k_blocks=n,root_contraction_cycles=(last-first)%(1<<48),max_abs_fp64_error=max(float(error[yy,xx].max()) for xx,yy in owners[group])))
    record=dict(case=case,schedule=kind,roots=timing,host_arm_and_ready_seconds=arm,host_start_through_result_audit_seconds=host);rows.append(record);print(json.dumps(record),flush=True)
   print(json.dumps(dict(phase='retention',case=case)),flush=True)
   np.testing.assert_array_equal(read('weights',128,16),data['weights'][case]);np.testing.assert_array_equal(read('scale',1),data['scales'][case].view(np.uint32));np.testing.assert_array_equal(read('raw',33),data['raw'][case])
 r=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope=meta['scope'],application=[w,h],blocks=blocks,fixture_sha256=meta['fixture_sha256'],cases=rows,clock_hz=None,all_inputs_and_weight_retained=True,
  timing_note='Each row root times its own full-K result: on-device activation packet encoding, weight decode, native dots and complete reduction including start-entry skew. Host receive arming/readiness is separate. Root timing does not include every sender completion callback; host full audit verifies those before rearm. Tree and chain use separately frozen FP32 schedules, not bitwise-equivalent regrouping; no full-M/model rate.')
 Path('result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,epochs=len(rows))),flush=True)
if __name__=='__main__':main()
