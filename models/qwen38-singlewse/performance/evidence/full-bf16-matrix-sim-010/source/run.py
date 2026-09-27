"""Complete matrix scope, exact intermediate gates and bounded host transfers."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());plan=json.loads(Path('region.json').read_text())
 assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
 workers=meta['workers'];w,h=meta['application'];runs=[];lines=[]
 # Source rank order reverses on the second physical row; transfers use x order.
 for y in range(h):
  line=sorted((r for r in workers if r['xy'][1]==y),key=lambda r:r['xy'][0])
  lines.append((line[0]['xy'][0],y,[r['rank'] for r in line]))
  for offset in range(0,len(line),16):
   group=line[offset:offset+16];xs=[r['xy'][0] for r in group]
   assert xs==list(range(xs[0],xs[0]+len(xs)));runs.append((xs[0],y,[r['rank'] for r in group]))
 records=[];captures={};chosen=meta['physical_cases'] if a.physical else meta['simulation_cases'];run_start=time.monotonic()
 def mark(phase,**extra):print(json.dumps(dict(phase=phase,wall_seconds=time.monotonic()-run_start,**extra)),flush=True)
 with runtime(a.physical) as (runner,dtype,order):
  names=['bank','target','scale0','packet','control','local_result','aggregate','gather','audit','ticks'];ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,value,x,y,width=1,height=1,bits=32):
   value=np.ascontiguousarray(value);runner.memcpy_h2d(ids[name],value.reshape(-1),x,y,width,height,value.size//(width*height),data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,n,x=0,y=0,width=1,height=1,bits=32):
   value=np.zeros(width*height*n,np.uint32);runner.memcpy_d2h(value,ids[name],x,y,width,height,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return value.reshape(height,width,n)
  def worker_read(name,n):
   value=np.zeros((960,n),np.uint32)
   for x,y,rr in lines:value[rr]=read(name,n,x,y,len(rr)).reshape(len(rr),n)
   return value
  def canon(value):return np.where(value==0,np.float32(0),value).astype(np.float32).view(np.uint32)
  print(json.dumps(dict(phase='upload_resident_banks',physical=a.physical)),flush=True);started=time.perf_counter()
  runner.launch('initialize_aliases',nonblock=False);mark('aliases_submitted')
  for x,y,rr in (runs if a.physical else lines):
   if a.physical:upload('bank',data['bank'][rr],x,y,len(rr))
   else:
    # The simulator executes BF16 case 0 only. Do not transfer unused FP8
    # smoke operands; full allocation and both native programs are unchanged.
    assert all(meta['cases'][c]['kind']=='bf16' for c in chosen)
    upload('target',data['target'][rr],x,y,len(rr));mark('target_upload_submitted',x=x,y=y,pes=len(rr))
  # Host command submission may finish before device execution in the simulator.
  # A returned D2H observation fences preceding initialization commands.
  np.testing.assert_array_equal(read('audit',1,0,0,w,h,bits=16),0);mark('initialization_observed')
  initialization=time.perf_counter()-started
  expected=np.zeros((h,w,12),np.uint32);expected[...,0]=1;expected[...,10]=1
  for x,y in plan['input_path']:expected[y,x,8]=1
  expected[0,0,[1,7,9]]=1
  # Rebuild node sizes/child counts independently as preorder intervals.
  sizes={};children={};pending=[(0,40)]
  while pending:
   first,size=pending.pop();sizes[first]=size;left=size//2;right=size-1-left;children[first]=(bool(left),bool(right))
   if left:pending.append((first+1,left))
   if right:pending.append((first+1+left,right))
  for r in workers:
   x,y=r['xy'];k=r['k'];expected[y,x,[1,2,5,7]]=1;expected[y,x,3:5]=children[k]
   if k==0:
    g=r['group'];steps=0
    while g%(2**(steps+1))==0 and g+2**steps<24:steps+=1
    pairs=min(2**steps,24-g);expected[y,x,6]=pairs-1;expected[y,x,7]=pairs
  for case in chosen:
   descriptor=meta['cases'][case];print(json.dumps(dict(phase='complete_matrix_epoch',case=case)),flush=True)
   control=np.zeros((h,w,2),np.uint32)
   for r in workers:control[r['xy'][1],r['xy'][0]]=[int(descriptor['kind']=='bf16'),descriptor['slot']]
   started=time.perf_counter();upload('control',control,0,0,w,h,16);upload('packet',data['stream'][case],0,0)
   print(json.dumps(dict(phase='arm',case=case)),flush=True);runner.launch('arm',nonblock=False)
   armed=read('audit',1,0,0,w,h,bits=16);np.testing.assert_array_equal(armed,1);mark('all_pes_armed',case=case)
   arming=time.perf_counter()-started;started=time.perf_counter();mark('start',case=case,upload_arm_seconds=arming)
   task=runner.launch('start',nonblock=True);mark('start_submitted',case=case)
   task_deadline=time.monotonic()+150
   while not runner.is_task_done(task):
    if time.monotonic()>task_deadline:
     Path('diagnostic.json').write_text(json.dumps(dict(case=case,phase='start_task_incomplete',wall_seconds=time.monotonic()-run_start))+'\n')
     raise TimeoutError('Start task completion deadline')
    time.sleep(.2)
   mark('start_task_done',case=case)
   previous=None;deadline=time.monotonic()+150
   while True:
    audit=read('audit',12,0,0,w,h,16)
    progress=[int(audit[...,i].sum()) for i in [1,2,5,6,7,8,10,11]]
    if progress!=previous:print(json.dumps(dict(phase='progress',case=case,received_dot_join_gather_sent_teardown_finished_errors=progress)),flush=True);previous=progress
    if np.all(audit[...,10]==1):break
    if time.monotonic()>deadline or np.any(audit[...,11]):
     Path('diagnostic.json').write_text(json.dumps(dict(case=case,application=[w,h],audit=audit.tolist()),separators=(',',':'))+'\n')
     raise RuntimeError('Matrix completion deadline or device error; frozen audit snapshot')
    time.sleep(.1)
   host=time.perf_counter()-started
   if not np.array_equal(audit,expected):
    bad=np.argwhere(np.any(audit!=expected,axis=2));Path('diagnostic.json').write_text(json.dumps(dict(case=case,counters=[dict(x=int(x),y=int(y),observed=audit[y,x].tolist(),expected=expected[y,x].tolist()) for y,x in bad[:64]]),indent=2)+'\n')
   np.testing.assert_array_equal(audit,expected)
   packets=worker_read('packet',65);np.testing.assert_array_equal(packets,data['stream'][case,np.arange(960)%40])
   local=worker_read('local_result',2).view(np.float32);np.testing.assert_array_equal(canon(local),canon(data['local'][case]))
   error=np.abs(local.astype(np.float64)-data['truth'][case]);assert np.isfinite(local).all() and np.all(error<=data['bounds'][case])
   aggregate=worker_read('aggregate',9)[:,6:9];tv=np.ascontiguousarray(aggregate[:,:2]).view(np.float32)
   np.testing.assert_array_equal(canon(tv),canon(data['trees'][case]));np.testing.assert_array_equal(aggregate[:,2],data['counts'])
   terror=np.abs(tv.astype(np.float64)-data['tree_truth'][case]);assert np.isfinite(tv).all() and np.all(terror<=data['tree_bounds'][case])
   # Streaming roots retain only their last forwarded pair. The source retains
   # the complete stream, which must contain every original row exactly in order.
   for group in range(24):
    r=workers[group*40];x,y=r['xy'];length=int(expected[y,x,7])
    observed=read('gather',2,x,y).view(np.float32).reshape(-1)
    np.testing.assert_array_equal(canon(observed),canon(data['trees'][case,(group+length-1)*40]))
   returned=read('gather',48).view(np.float32).reshape(48);wanted=data['trees'][case,::40].reshape(48)
   np.testing.assert_array_equal(canon(returned),canon(wanted))
   output_error=np.abs(returned.astype(np.float64)-data['tree_truth'][case,::40].reshape(48));assert np.all(output_error<=data['tree_bounds'][case,::40].reshape(48))
   ticks=read('ticks',6,bits=16).reshape(6);cycles=(sum(int(ticks[j+3])<<(16*j) for j in range(3))-sum(int(ticks[j])<<(16*j) for j in range(3)))%(1<<48);assert 0<cycles<10000000
   captures['outputs_'+str(case)]=returned;captures['local_'+str(case)]=local;captures['aggregate_'+str(case)]=aggregate;captures['ticks_'+str(case)]=ticks
   if descriptor['replay_of'] is not None:np.testing.assert_array_equal(canon(returned),canon(captures['outputs_'+str(descriptor['replay_of'])]))
   record=dict(descriptor,source_send_to_all_outputs_return_cycles=cycles,max_local_fp64_error=float(error.max()),max_subtree_fp64_error=float(terror.max()),max_output_fp64_error=float(output_error.max()),host_upload_arm_seconds=arming,host_start_seconds=host)
   records.append(record);print(json.dumps(record),flush=True)
  print(json.dumps(dict(phase='bank_retention',full_background=a.physical)),flush=True)
  for x,y,rr in (runs if a.physical else lines):
   if a.physical:np.testing.assert_array_equal(read('bank',8814,x,y,len(rr)).reshape(len(rr),8814),data['bank'][rr])
   else:
    np.testing.assert_array_equal(read('target',128,x,y,len(rr)).reshape(len(rr),128),data['target'][rr])
  np.savez('actual.npz',**captures)
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,application=[w,h],scope=meta['scope'],fixture_sha256=meta['fixture_sha256'],cases=records,
  complete_original_matrix_shape=[48,5120],original_target_weights_retained=True,all_loaded_weights_retained=True,full_background_retained=a.physical,simulator_payload='Complete original BF16 target slot only; unused FP8/background zero initialized' if not a.physical else None,
  all_native_dots_subtrees_gathers_and_outputs_exact=True,all_packet_and_callback_counters_exact=True,host_initialization_seconds=initialization,
  timing_note='Same source clock from2600-word input send to all48 output values returned. Includes input,960 native tiles,24 K40 trees,ordered vector gather and return. Host completion snapshots may overlap and are not subtracted. Excludes host arm and final retention. Independent host operands, no model rate.',clock_hz=None,full_model_speed_target_achieved=False)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical)),flush=True)
if __name__=='__main__':main()
