"""All753 real-width recurrent ports: source/consumer fences and full state."""
import argparse,json,time
from pathlib import Path
import numpy as np
from backend import runtime,file_sha256

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--physical',action='store_true');parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
 if args.physical and args.smoke:raise ValueError('Physical qualification requires six continuous and two reset positions')
 meta=json.loads(Path('fixture.json').read_text());plan=json.loads(Path('probe-plan.json').read_text());workers=plan['workers'];width,height=plan['application']
 if args.physical and (len(workers)!=753 or [width,height]!=[251,6]):raise ValueError('All original slices required on physical hardware')
 if file_sha256(Path('fixture.npz'))!=meta['fixture_sha256'] or not meta['passed']:raise ValueError('Frozen fixture')
 data=dict(np.load('fixture.npz',allow_pickle=False));started=time.monotonic();records=[];captures={};replay_values={}
 def progress(phase,**kw):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-started,**kw)),flush=True)
 progress('runtime_load')
 with runtime(args.physical) as (runner,dtype,order):
  progress('runtime_loaded')
  ids={n:runner.get_id(n) for n in ['state','status','wire','frames']};options=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
  def read(name,count,worker=True):
   result=np.zeros((len(workers),count),np.uint32)
   for group in range(height//2):
    target=result[group*width:(group+1)*width]
    runner.memcpy_d2h(target.reshape(-1),ids[name],0,2*group+int(worker),width,1,count,**options)
   return result
  def write_wire(packets):
   for group in range(height//2):
    source=np.ascontiguousarray(packets[group*width:(group+1)*width],np.uint32)
    runner.memcpy_h2d(ids['wire'],source.reshape(-1),0,2*group,width,1,387,**options)
  def launch(name):runner.launch(name,nonblock=False)
  for replay,length in ([(False,1)] if args.smoke else [(False,6),(True,2)]):
   progress('reset',replay=replay);launch('clear');progress('reset_sent');np.testing.assert_array_equal(read('state',4096),0);progress('reset_checked')
   for position in range(length):
    token=position+1;progress('position',token=token,replay=replay)
    packets=np.zeros((len(workers),387),np.uint32);packets[:,0]=token
    for i,w in enumerate(workers):packets[i,1:]=data[f'packet_{position}'][w['head']].view(np.uint32)
    write_wire(packets);launch('begin');launch('exchange')
    # exchange unblocks only after each source has sent its full packet AND
    # received every return word. Worker local-send idle is checked separately.
    deadline=time.monotonic()+15
    while True:
     launch('inspect');status=read('status',3)
     if np.all(status[:,1]==0):break
     if time.monotonic()>deadline:raise ValueError('Worker local retirement timeout')
    np.testing.assert_array_equal(status[:,0],token)
    np.testing.assert_array_equal(status[:,2],[w['columns'] for w in workers])
    source_status=read('status',3,False);np.testing.assert_array_equal(source_status[:,:2],1);np.testing.assert_array_equal(source_status[:,2],token)
    frames=read('frames',80,False);states=read('state',4096)
    outputs=np.zeros((48,128),np.uint16);state=np.zeros((48,128,128),np.float32);seen=np.zeros((48,128),np.bool_)
    for i,w in enumerate(workers):
     h,first,n=w['head'],w['first'],w['columns'];frame=frames[i,:5*n//2].reshape(-1,5)
     np.testing.assert_array_equal(frame[:,0],token);np.testing.assert_array_equal(frame[:,1],h*128+first+np.arange(0,n,2));np.testing.assert_array_equal(frame[:,2],2);np.testing.assert_array_equal(frame[:,4],5)
     if np.any(seen[h,first:first+n]):raise ValueError('Duplicated value slice')
     seen[h,first:first+n]=True;packed=frame[:,3];outputs[h,first:first+n:2]=(packed&65535).astype(np.uint16);outputs[h,first+1:first+n:2]=(packed>>16).astype(np.uint16)
     state[h,:,first:first+n]=states[i,:128*n].copy().view(np.float32).reshape(128,n)
     np.testing.assert_array_equal(states[i,128*n:],0)
    if args.physical and not np.all(seen):raise ValueError('Incomplete original heads')
    values=(outputs.astype(np.uint32)<<16).view(np.float32).astype(np.float64);difference=np.abs(state.astype(np.float64)-data[f'state_{position}'])
    covered=np.broadcast_to(seen[:,None,:],state.shape)
    state_pass=bool(np.isfinite(state).all() and np.all(difference[covered]<=data[f'bound_{position}'][covered]))
    output_pass=bool(np.isfinite(values).all() and np.all(values[seen]>=data[f'lower_{position}'][seen]) and np.all(values[seen]<=data[f'upper_{position}'][seen]))
    captures[f'state_{int(replay)}_{position}']=state;captures[f'output_{int(replay)}_{position}']=outputs
    if not state_pass or not output_pass:
     np.savez('rejected.npz',state=state,output=outputs,difference=difference,bound=data[f'bound_{position}']);raise ValueError('Frozen recurrent numerical bounds failed: state='+str(state_pass)+' output='+str(output_pass))
    if replay:
     for actual,old in zip((state,outputs),replay_values[position]):np.testing.assert_array_equal(actual,old)
    elif position<2:replay_values[position]=(state.copy(),outputs.copy())
    records.append(dict(position=position,replay=replay,state_interval_pass=state_pass,output_interval_pass=output_pass,paired_frames=int(seen.sum())//2,state_elements=int(covered.sum()),exact_reset_replay=replay,transport_drained=True))
    np.savez('actual.npz',**captures);progress('checked',**records[-1])
 result=dict(passed=True,physical=args.physical,normal_stop=True,heads=len({w['head'] for w in workers}),workers=len(workers),full_original_value_coverage=bool(np.all(seen)),positions=1 if args.smoke else 6,reset_replay_positions=0 if args.smoke else 2,records=records,fixture_sha256=meta['fixture_sha256'],scope=meta['scope'] if len(workers)==753 else 'Selected8/32-column original slices: exact paired-port transport/math smoke only; complete physical qualification remains required.',full_model_speed_target_achieved=False,seconds=time.monotonic()-started)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
