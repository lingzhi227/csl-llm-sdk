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
 wx=np.array([r['xy'][0] for r in workers]);wy=np.array([r['xy'][1] for r in workers])
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
  names=['bank','target','packet','control','local_result','aggregate','gather','audit','ticks'];ids={n:runner.get_id(n) for n in names};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,value,x,y,width=1,height=1,bits=32):
   value=np.ascontiguousarray(value);runner.memcpy_h2d(ids[name],value.reshape(-1),x,y,width,height,value.size//(width*height),data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,n,x=0,y=0,width=1,height=1,bits=32):
   value=np.zeros(width*height*n,np.uint32);runner.memcpy_d2h(value,ids[name],x,y,width,height,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return value.reshape(height,width,n)
  def worker_read(name,n):
   if name in ['local_result','aggregate','gather']:
    # These exports have a sufficient uniform extent on every PE. A single
    # rectangle uses both memcpy channels without serial per-row commands.
    return read(name,n,0,0,w,h)[wy,wx].copy()
   value=np.zeros((960,n),np.uint32)
   for x,y,rr in lines:value[rr]=read(name,n,x,y,len(rr)).reshape(len(rr),n)
   return value
  def canon(value):return np.where(value==0,np.float32(0),value).astype(np.float32).view(np.uint32)
  payload_checks=[]
  def exact_payload(expected,kind,bad_rank):
   # All scratch/subtree results must already have been captured. Compare every
   # original u32 on device, prove a one-bit negative control, then restore only
   # the deliberately changed expected PE and demand zero mismatches everywhere.
   gold=np.ascontiguousarray(expected,dtype=np.uint32);n=gold.shape[1];assert gold.shape==(960,n) and n in [65,128]
   poisoned=gold.copy();poisoned[bad_rank,-1]^=np.uint32(1)
   box=np.zeros((h,w,n),np.uint32);box[wy,wx]=poisoned
   mark('payload_negative_upload',kind=kind,words=960*n)
   upload('aggregate',box,0,0,w,h)
   runner.launch('check_payload',np.uint16(kind),nonblock=False)
   mark('payload_negative_read',kind=kind)
   counts=worker_read('local_result',2);wanted=np.zeros((960,2),np.uint32);wanted[:,0]=n;wanted[bad_rank,1]=1
   np.testing.assert_array_equal(counts,wanted)
   x,y=workers[bad_rank]['xy'];upload('aggregate',gold[bad_rank],x,y)
   runner.launch('check_payload',np.uint16(kind),nonblock=False)
   restored=worker_read('local_result',2);wanted[bad_rank,1]=0;np.testing.assert_array_equal(restored,wanted)
   entry=dict(kind=kind,compared_words=960*n,negative_control_rank=bad_rank,negative_control_word=n-1,negative_control_mismatches=1,restored_mismatches=0,expected_sha256=hashlib.sha256(gold.tobytes()).hexdigest())
   payload_checks.append(entry);mark('exact_payload_checked',**entry)
  print(json.dumps(dict(phase='upload_resident_banks',physical=a.physical)),flush=True);started=time.perf_counter()
  runner.launch('initialize_aliases',nonblock=False);mark('aliases_submitted')
  if a.physical:
   for x,y,rr in runs:upload('bank',data['bank'][rr],x,y,len(rr))
  else:
   # The simulator executes BF16 case 0 only. Non-worker target aliases point
   # to an explicitly allocated 128-word dummy region; no out-of-bounds copy.
   assert all(meta['cases'][c]['kind']=='bf16' for c in chosen)
   box=np.zeros((h,w,128),np.uint32);box[wy,wx]=data['target']
   upload('target',box,0,0,w,h);mark('target_upload_submitted',pes=960,rectangle_pes=w*h)
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
    if np.all(audit[...,10]==1):
     # The first snapshot may straddle completion within a PE. Once every PE
     # has finished, take one stable snapshot for the exact callback gate.
     audit=read('audit',12,0,0,w,h,16);break
    if time.monotonic()>deadline or np.any(audit[...,11]):
     Path('diagnostic.json').write_text(json.dumps(dict(case=case,application=[w,h],audit=audit.tolist()),separators=(',',':'))+'\n')
     raise RuntimeError('Matrix completion deadline or device error; frozen audit snapshot')
    time.sleep(.1)
   host=time.perf_counter()-started
   if not np.array_equal(audit,expected):
    bad=np.argwhere(np.any(audit!=expected,axis=2));Path('diagnostic.json').write_text(json.dumps(dict(case=case,counters=[dict(x=int(x),y=int(y),observed=audit[y,x].tolist(),expected=expected[y,x].tolist()) for y,x in bad[:64]]),indent=2)+'\n')
   np.testing.assert_array_equal(audit,expected)
   mark('completion_counters_checked',case=case)
   if a.physical:
    packets=worker_read('packet',65);np.testing.assert_array_equal(packets,data['stream'][case,np.arange(960)%40]);mark('all_input_packets_checked',case=case)
   local=worker_read('local_result',2).view(np.float32);np.testing.assert_array_equal(canon(local),canon(data['local'][case]))
   error=np.abs(local.astype(np.float64)-data['truth'][case]);assert np.isfinite(local).all() and np.all(error<=data['bounds'][case])
   mark('all_native_dots_checked',case=case)
   aggregate=worker_read('aggregate',9)[:,6:9];tv=np.ascontiguousarray(aggregate[:,:2]).view(np.float32)
   np.testing.assert_array_equal(canon(tv),canon(data['trees'][case]));np.testing.assert_array_equal(aggregate[:,2],data['counts'])
   terror=np.abs(tv.astype(np.float64)-data['tree_truth'][case]);assert np.isfinite(tv).all() and np.all(terror<=data['tree_bounds'][case])
   mark('all_subtrees_checked',case=case)
   # Streaming roots retain only their last forwarded pair. The source retains
   # the complete stream, which must contain every original row exactly in order.
   gathers=worker_read('gather',2).view(np.float32)
   np.testing.assert_array_equal(gathers[np.arange(960)%40!=0],0)
   for group in range(24):
    r=workers[group*40];x,y=r['xy'];length=int(expected[y,x,7])
    observed=gathers[group*40]
    np.testing.assert_array_equal(canon(observed),canon(data['trees'][case,(group+length-1)*40]))
   returned=read('gather',48).view(np.float32).reshape(48);wanted=data['trees'][case,::40].reshape(48)
   np.testing.assert_array_equal(canon(returned),canon(wanted))
   output_error=np.abs(returned.astype(np.float64)-data['tree_truth'][case,::40].reshape(48));assert np.all(output_error<=data['tree_bounds'][case,::40].reshape(48))
   ticks=read('ticks',6,bits=16).reshape(6);cycles=(sum(int(ticks[j+3])<<(16*j) for j in range(3))-sum(int(ticks[j])<<(16*j) for j in range(3)))%(1<<48);assert 0<cycles<10000000
   captures['outputs_'+str(case)]=returned;captures['local_'+str(case)]=local;captures['aggregate_'+str(case)]=aggregate;captures['ticks_'+str(case)]=ticks
   mark('all_outputs_checked',case=case,cycles=cycles)
   if not a.physical:exact_payload(data['stream'][case,np.arange(960)%40],0,0)
   if descriptor['replay_of'] is not None:np.testing.assert_array_equal(canon(returned),canon(captures['outputs_'+str(descriptor['replay_of'])]))
   record=dict(descriptor,source_send_to_all_outputs_return_cycles=cycles,max_local_fp64_error=float(error.max()),max_subtree_fp64_error=float(terror.max()),max_output_fp64_error=float(output_error.max()),host_upload_arm_seconds=arming,host_start_seconds=host)
   records.append(record);print(json.dumps(record),flush=True)
  print(json.dumps(dict(phase='bank_retention',full_background=a.physical)),flush=True)
  if a.physical:
   for x,y,rr in runs:np.testing.assert_array_equal(read('bank',8814,x,y,len(rr)).reshape(len(rr),8814),data['bank'][rr])
  else:exact_payload(data['target'],1,959)
  np.savez('actual.npz',**captures)
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,application=[w,h],scope=meta['scope'],fixture_sha256=meta['fixture_sha256'],cases=records,
  complete_original_matrix_shape=[48,5120],original_target_weights_retained=True,all_loaded_weights_retained=True,full_background_retained=a.physical,simulator_payload='Complete original BF16 target slot only; unused FP8/background zero initialized' if not a.physical else None,
  all_native_dots_subtrees_gathers_and_outputs_exact=True,all_packet_and_callback_counters_exact=True,host_initialization_seconds=initialization,
  exact_device_payload_checks=payload_checks,validation_note='Simulation compares every packet/target word on device against separately uploaded immutable originals in dead scratch, with deliberate one-bit negative controls and restoration. All local/subtree/final numerical values are independently checked on host. Physical run retains full packet and full bank D2H comparisons.',
  timing_note='Same source clock from2600-word input send to all48 output values returned. Includes input,960 native tiles,24 K40 trees,ordered vector gather and return. Host completion snapshots may overlap and are not subtracted. Excludes host arm and final retention. Independent host operands, no model rate.',clock_hz=None,full_model_speed_target_achieved=False)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical)),flush=True)
if __name__=='__main__':main()
