"""Verify an automatically connected device graph at component boundaries."""
import argparse,json,time
from pathlib import Path
import numpy as np
from backend import runtime,file_sha256
from reference.gdn_columns_oracle import step
from reference.frontend_oracle import expand,gated_interval

def timing_record(frontend,workers):
 def unpack(a):
  a=np.asarray(a,dtype=np.uint64).reshape(len(a),-1,3)
  return a[:,:,0]+(a[:,:,1]<<np.uint64(16))+(a[:,:,2]<<np.uint64(32))
 def delta(a,b):return (a-b)&np.uint64((1<<48)-1)
 f=unpack(frontend);w=unpack(workers)
 relative=delta(w,w[:,:1])
 if not np.all(relative[:,1:]>=relative[:,:-1]):raise ValueError('Worker timestamp order')
 span=delta(f[:,6],f[:,0]);heads=delta(f[:,3:6],f[:,:3])
 if np.any(span==0) or np.any(heads==0) or np.any(delta(f[:,6:7],f[:,3:6])>span[:,None]):raise ValueError('Frontend timestamp order')
 return dict(units='PE counter cycles',scope='Instrumented component; original projected inputs supplied at diagnostic source; forced delayed callback and capture overhead included. No full-layer/model TPS.',
             frontend_first_launch_to_drain=span.tolist(),frontend_head_launch_to_gated_output=heads.tolist(),
             worker_key_update_min=int(np.min(delta(w[:,3],w[:,2]))),worker_key_update_max=int(np.max(delta(w[:,3],w[:,2]))),
             worker_query_reduce_min=int(np.min(delta(w[:,5],w[:,4]))),worker_query_reduce_max=int(np.max(delta(w[:,5],w[:,4]))),
             worker_return_drain_min=int(np.min(delta(w[:,7],w[:,5]))),worker_return_drain_max=int(np.max(delta(w[:,7],w[:,5]))))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--physical',action='store_true');args=parser.parse_args()
 plan=json.loads(Path('probe-plan.json').read_text());meta=json.loads(Path('fixture.json').read_text())
 if args.physical and (args.smoke or not plan['full_original_value_coverage']):raise ValueError('Full connected profile required')
 if file_sha256(Path('fixture.npz'))!=meta['fixture_sha256']:raise ValueError('Frozen original inputs')
 data=dict(np.load('fixture.npz',allow_pickle=False));groups=plan['groups'];width=len(groups);workers=plan['workers']
 started=time.monotonic();records=[];captures={};prior={}
 def progress(phase,**kw):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-started,**kw)),flush=True)
 progress('runtime_load')
 with runtime(args.physical) as (runner,dtype,order):
  progress('runtime_loaded');ids={n:runner.get_id(n) for n in ['wire','weights','history','gain','parameters','packet','core','output','audit','status','state','ticks']}
  options=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def write(name,value,y=1,bits=16):
   a=np.ascontiguousarray(value,np.uint32).reshape(width,-1)
   for i,(xx,yy) in enumerate(plan['sources' if y==0 else 'frontends']):
    runner.memcpy_h2d(ids[name],a[i],xx,yy,1,1,a.shape[1],data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**options)
  def read(name,count,y=None,bits=32,x=None,w=1,h=1):
   if x is None:return np.concatenate([read(name,count,y=yy,bits=bits,x=xx) for xx,yy in plan['frontends']])
   progress('read_begin',name=name,count=count,x=x,y=y,width=w,height=h)
   a=np.zeros((h,w,count),np.uint32)
   runner.memcpy_d2h(a.reshape(-1),ids[name],x,y,w,h,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**options)
   progress('read_end',name=name,count=count,x=x,y=y,width=w,height=h)
   a=a.reshape(h*w,count);return a if bits==32 else (a&65535).astype(np.uint16)
  def worker_read(name,count,g,bits=32):
   local=[v for v in workers if v['head']//3==g];values={}
   for yy in sorted({v['diagnostic_pe'][1] for v in local}):
    xs=sorted(v['diagnostic_pe'][0] for v in local if v['diagnostic_pe'][1]==yy)
    if xs!=list(range(xs[0],xs[-1]+1)):raise ValueError('Diagnostic contiguous worker interval')
    # Observe every live recurrent value, without transferring the diagnostic
    # 4096-word bank's unused tail for an eight-column slice.
    row={v['diagnostic_pe'][0]:v for v in local if v['diagnostic_pe'][1]==yy}
    start=0
    while start<len(xs):
     words=128*row[xs[start]]['columns'] if name=='state' else count
     end=start+1
     while end<len(xs) and (128*row[xs[end]]['columns'] if name=='state' else count)==words:end+=1
     block=read(name,words,y=yy,x=xs[start],w=end-start,bits=bits)
     for i,xx in enumerate(xs[start:end]):
      values[(xx,yy)]=np.zeros(count,np.uint32);values[(xx,yy)][:words]=block[i]
     start=end
   return local,np.stack([values[tuple(v['diagnostic_pe'])] for v in local])
  def launch(name,*values):runner.launch(name,*values,nonblock=False)
  for name in ['weights','gain','parameters']:write(name,data[name])
  launch('clear');launch('arm');launch('inspect')
  # SDK launch completion is not a whole-graph admission acknowledgement.
  # Observe each real receiver before the diagnostic source may inject data.
  for x,g in enumerate(groups):
   _,s=worker_read('status',4,g);np.testing.assert_array_equal(s[:,1],0);np.testing.assert_array_equal(s[:,3],1)
  for replay,length in ([(False,2)] if args.smoke else [(False,6),(True,2)]):
   if replay:launch('clear')
   states=np.zeros((width,3,128,128),np.float64);errors=np.zeros_like(states)
   for position in range(length):
    token=position+1;progress('token_begin',token=token,replay=replay);launch('begin',np.uint32(token));launch('inspect')
    admitted=read('status',16);progress('frontend_admission',status=admitted.tolist());np.testing.assert_array_equal(admitted[:,0],token)
    for start in range(0,518,64):
     buffer=np.zeros((width,321),np.uint32)
     for x in range(width):
      chunk=data[f'projected_{position}_{x}'][start:start+64];buffer[x,0]=len(chunk);buffer[x,1:1+chunk.size]=chunk.reshape(-1)
     write('wire',buffer,y=0,bits=32);launch('send');progress('source_batch_enqueued',start=start)
    launch('await_done');launch('inspect');status=read('status',16)
    expected=np.array([[token,7,7,7,7,0,0,1,710,710,sum(w['head']//3==g for w in workers),192,token,3,0,sum(w['head']//3==g for w in workers)] for g in groups],np.uint32)
    np.testing.assert_array_equal(status,expected);np.testing.assert_array_equal(read('audit',8),np.tile([token,710,320,192,3,3,3,1],(width,1)))
    progress('device_graph_drained',token=token,replay=replay)
    # Each SDK row has its own command stream. The inspect queued behind the
    # frontend await on row0 can execute earlier on a worker's different row.
    # First observe actual frontend drain, THEN issue a fresh worker snapshot.
    launch('inspect')
    for g in groups:
     local,ws=worker_read('status',3,g)
     for retry in range(3):
      if np.all(ws[:,1]==0):break
      launch('inspect');local,ws=worker_read('status',3,g)
     np.testing.assert_array_equal(ws[:,0],token);np.testing.assert_array_equal(ws[:,1],0)
     np.testing.assert_array_equal(ws[:,2],[w['columns'] for w in local])
    progress('worker_local_drained',token=token,replay=replay)
    frontend_ticks=read('ticks',21,bits=16)
    packet=read('packet',1161).reshape(width,3,387);np.testing.assert_array_equal(packet[:,:,0],token)
    p=packet[:,:,1:].copy().view(np.float32)
    old=np.concatenate((p[:,:,258:],p[:,:,130:258],p[:,:,2:130],p[:,:,:2]),axis=2)
    fe=np.abs(old.astype(np.float64)-data[f'packet_{position}'])
    if not np.isfinite(old).all() or not np.all(fe<=data[f'bound_{position}']):raise ValueError('Independent frontend interval')
    np.testing.assert_array_equal(packet[:,0,131:],packet[:,1,131:]);np.testing.assert_array_equal(packet[:,0,131:],packet[:,2,131:])
    history=read('history',1920,bits=16).reshape(width,640,3);np.testing.assert_array_equal(history,data[f'history_{position}'])
    core=read('core',384,bits=16).reshape(width,3,128);output=read('output',384,bits=16).reshape(width,3,128)
    actual=np.zeros((width,3,128,128),np.float32);ratio=0.;relative=0.
    worker_ticks=[]
    for x,g in enumerate(groups):
     _,wt=worker_read('ticks',24,g,bits=16);worker_ticks.extend(wt)
     local,bank=worker_read('state',4096,g);bank=bank.view(np.float32)
     for i,w in enumerate(local):actual[x,w['head']%3,:,w['first']:w['first']+w['columns']]=bank[i,:128*w['columns']].reshape(128,w['columns'])
     for h in range(3):
      states[x,h],errors[x,h],lo,hi,_=step(states[x,h],errors[x,h],p[x,h])
      error=np.abs(actual[x,h].astype(np.float64)-states[x,h])
      if not np.isfinite(actual[x,h]).all() or not np.all(error<=errors[x,h]):raise ValueError('Independent persistent recurrent state interval')
      ratio=max(ratio,float(np.max(error/np.maximum(errors[x,h],1e-300))))
      relative=max(relative,float(np.linalg.norm(error)/max(np.linalg.norm(states[x,h]),1e-30)))
      ce=expand(core[x,h]);oe=expand(output[x,h])
      if not np.isfinite(ce).all() or not np.all((ce>=lo)&(ce<=hi)):raise ValueError('Independent core return interval')
      lo,hi=gated_interval(core[x,h],data[f'z_{position}'][x,128*h:128*(h+1)],data['gain'][x])
      if not np.isfinite(oe).all() or not np.all((oe>=lo)&(oe<=hi)):raise ValueError('Independent gated output interval')
    timing=timing_record(frontend_ticks,np.asarray(worker_ticks))
    captures[f'frontend_ticks_{int(replay)}_{position}']=frontend_ticks
    captures[f'worker_ticks_{int(replay)}_{position}']=np.asarray(worker_ticks,np.uint16)
    values=dict(packet=packet,history=history,core=core,output=output,state=actual)
    if replay:
     for name,a in values.items():np.testing.assert_array_equal(a.view(np.uint8),prior[position][name].view(np.uint8))
    elif position<2:prior[position]={n:a.copy() for n,a in values.items()}
    captures.update({f'{n}_{int(replay)}_{position}':a for n,a in values.items()});np.savez('actual.npz',**captures)
    record=dict(position=position,replay=replay,frontend_interval=True,recurrent_state_interval=True,core_interval=True,gated_interval=True,
                exact_history=True,all_return_markers=True,delayed_source_callback=True,timing=timing,maximum_state_error_bound_ratio=ratio,maximum_state_relative_l2=relative)
    records.append(record);progress('position_verified',**record)
  for name in ['weights','gain','parameters']:np.testing.assert_array_equal(read(name,data[name].reshape(width,-1).shape[1],bits=16),data[name].reshape(width,-1))
 result=dict(passed=True,physical=args.physical,normal_stop=True,groups=groups,workers=len(workers),positions=2 if args.smoke else 6,reset_replay_positions=0 if args.smoke else 2,
             actual_frontend_execution=True,actual_gdn_execution=True,host_injected_recurrent_results=False,full_original_value_coverage=plan['full_original_value_coverage'],
             records=records,fixture_sha256=meta['fixture_sha256'],scope=plan['scope'],full_model_speed_target_achieved=False)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
