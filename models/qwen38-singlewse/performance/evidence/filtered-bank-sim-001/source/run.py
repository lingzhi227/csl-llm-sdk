"""Complete two-stage window checks, typed dots and original-bank readback."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
 workers=meta['workers'];chosen=meta['physical_cases'] if a.physical else meta['simulation_cases'];records=[];captures={}
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','scale','bfweights','control','packet','local_result','audit','ticks']};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,value,x,y,w,h,bits=32):
   value=np.ascontiguousarray(value);n=value.size//(w*h);runner.memcpy_h2d(ids[name],value.reshape(-1),x,y,w,h,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,n,x=0,y=0,w=4,h=5,bits=32):
   value=np.zeros(w*h*n,np.uint32);runner.memcpy_d2h(value,ids[name],x,y,w,h,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return value.reshape(h,w,n)
  def canon(v):return np.where(v==0,np.float32(0),v).astype(np.float32).view(np.uint32)
  start=time.perf_counter();print(json.dumps(dict(phase='upload_resident_banks')),flush=True)
  for y in range(1,5):
   for name in ['weights','scale','bfweights']:
    upload(name,np.stack([data[name+'_'+str((y-1)*3+x)] for x in range(3)]),1,y,3,1)
  initialization=time.perf_counter()-start
  expected=np.zeros((5,4,9),np.uint32);expected[...,0]=1;expected[...,7]=1
  expected[0,0,[3,4,5]]=1;expected[0,1:,[1,3,4,5,6]]=1
  expected[1:,1:,1:3]=1;expected[1:,1:,6]=1
  for case in chosen:
   print(json.dumps(dict(phase='filtered_epoch',case=case)),flush=True);start=time.perf_counter()
   upload('control',data['control'][case].astype(np.uint32),0,0,4,5,16);period=meta['cases'][case]['period_words']
   upload('packet',data['stream'][case,:period],0,0,1,1);runner.launch('arm',nonblock=False)
   arming=time.perf_counter()-start;start=time.perf_counter();runner.launch('start',nonblock=False);host=time.perf_counter()-start
   np.testing.assert_array_equal(read('audit',9),expected)
   packet=read('packet',65,1,1,3,4);np.testing.assert_array_equal(packet,data['packets'][case])
   relay=read('packet',130,1,0,3,1)
   for x,offset in enumerate(meta['cases'][case]['pair_offsets_words']):np.testing.assert_array_equal(relay[0,x],data['stream'][case,offset:offset+130])
   actual=read('local_result',2,1,1,3,4).view(np.float32);np.testing.assert_array_equal(canon(actual),canon(data['local'][case]))
   err=np.abs(actual.astype(np.float64)-data['truth'][case]);assert np.isfinite(actual).all() and np.all(err<=data['bounds'][case])
   ticks=read('ticks',6,1,1,3,4,bits=16).reshape(12,6)
   cycles=[(sum(int(t[3+j])<<(16*j) for j in range(3))-sum(int(t[j])<<(16*j) for j in range(3)))%(1<<48) for t in ticks]
   assert all(0<v<10000000 for v in cycles)
   captures['result_'+str(case)]=actual.copy();captures['packets_'+str(case)]=packet;captures['ticks_'+str(case)]=ticks
   source=meta['cases'][case]['replay_of']
   if source is not None:np.testing.assert_array_equal(canon(actual),canon(captures['result_'+str(source)]))
   record=dict(meta['cases'][case],max_abs_fp64_error=float(err.max()),received_to_dot_done_cycles=cycles,host_upload_and_arm_seconds=arming,host_start_seconds=host)
   records.append(record);print(json.dumps(record),flush=True)
  print(json.dumps(dict(phase='full_bank_retention')),flush=True)
  for y in range(1,5):
   for name in ['weights','scale','bfweights']:
    wanted=np.stack([data[name+'_'+str((y-1)*3+x)] for x in range(3)])
    if name=='scale':wanted=wanted.view(np.uint32)
    np.testing.assert_array_equal(read(name,wanted.shape[-1],1,y,3,1)[0],wanted)
  np.savez('actual.npz',**captures)
 result=dict(passed=True,normal_stop=True,physical=a.physical,application=[4,5],full_model=False,scope=meta['scope'],
  fixture_sha256=meta['fixture_sha256'],cases=records,all_weights_retained=True,original_weights=True,weight_uploads=1,
  counter_windows_exact=True,all_packet_and_teardown_counters_exact=True,host_initialization_seconds=initialization,
  timing_note='Per worker receipt callback to local native dot completion only; excludes upstream delivery. Host start timing includes host/SDK latency. No cross-PE clock subtraction, end-to-end cycle or model rate claim.',clock_hz=None,full_model_speed_target_achieved=False)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical)),flush=True)
if __name__=='__main__':main()
