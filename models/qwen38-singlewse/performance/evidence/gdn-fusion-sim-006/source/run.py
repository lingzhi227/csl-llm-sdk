"""Verify an automatically connected device graph at component boundaries."""
import argparse,json,time
from pathlib import Path
import numpy as np
from backend import runtime,file_sha256
from reference.gdn_columns_oracle import step
from reference.frontend_oracle import expand,gated_interval

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
  progress('runtime_loaded');ids={n:runner.get_id(n) for n in ['wire','weights','history','gain','parameters','packet','core','output','audit','status','state']}
  options=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def write(name,value,y=1,bits=16):
   a=np.ascontiguousarray(value,np.uint32).reshape(width,-1)
   runner.memcpy_h2d(ids[name],a.reshape(-1),0,y,width,1,a.shape[1],data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**options)
  def read(name,count,y=1,bits=32,x=0,w=width,h=1):
   a=np.zeros((h,w,count),np.uint32)
   runner.memcpy_d2h(a.reshape(-1),ids[name],x,y,w,h,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**options)
   a=a.reshape(h*w,count);return a if bits==32 else (a&65535).astype(np.uint16)
  def launch(name,*values):runner.launch(name,*values,nonblock=False)
  for name in ['weights','gain','parameters']:write(name,data[name])
  launch('clear');launch('arm');launch('inspect')
  # SDK launch completion is not a whole-graph admission acknowledgement.
  # Observe each real receiver before the diagnostic source may inject data.
  for x,g in enumerate(groups):
   n=sum(w['head']//3==g for w in workers)
   np.testing.assert_array_equal(read('status',3,y=2,x=x,w=1,h=n)[:,1],1)
  for replay,length in ([(False,2)] if args.smoke else [(False,6),(True,2)]):
   if replay:launch('clear')
   states=np.zeros((width,3,128,128),np.float64);errors=np.zeros_like(states)
   for position in range(length):
    token=position+1;progress('token_begin',token=token,replay=replay);launch('begin',np.uint32(token));launch('inspect')
    np.testing.assert_array_equal(read('status',16)[:,0],token)
    for start in range(0,518,64):
     buffer=np.zeros((width,321),np.uint32)
     for x in range(width):
      chunk=data[f'projected_{position}_{x}'][start:start+64];buffer[x,0]=len(chunk);buffer[x,1:1+chunk.size]=chunk.reshape(-1)
     write('wire',buffer,y=0,bits=32);launch('send')
    launch('await_done');launch('inspect');status=read('status',16)
    expected=np.array([[token,7,7,7,7,0,0,1,710,710,sum(w['head']//3==g for w in workers),192,token,3,0,sum(w['head']//3==g for w in workers)] for g in groups],np.uint32)
    np.testing.assert_array_equal(status,expected);np.testing.assert_array_equal(read('audit',8),np.tile([token,710,320,192,3,3,3,1],(width,1)))
    progress('device_graph_drained',token=token,replay=replay)
    packet=read('packet',1161).reshape(width,3,387);np.testing.assert_array_equal(packet[:,:,0],token)
    p=packet[:,:,1:].copy().view(np.float32)
    old=np.concatenate((p[:,:,258:],p[:,:,130:258],p[:,:,2:130],p[:,:,:2]),axis=2)
    fe=np.abs(old.astype(np.float64)-data[f'packet_{position}'])
    if not np.isfinite(old).all() or not np.all(fe<=data[f'bound_{position}']):raise ValueError('Independent frontend interval')
    np.testing.assert_array_equal(packet[:,0,131:],packet[:,1,131:]);np.testing.assert_array_equal(packet[:,0,131:],packet[:,2,131:])
    history=read('history',1920,bits=16).reshape(width,640,3);np.testing.assert_array_equal(history,data[f'history_{position}'])
    core=read('core',384,bits=16).reshape(width,3,128);output=read('output',384,bits=16).reshape(width,3,128)
    actual=np.zeros((width,3,128,128),np.float32);ratio=0.;relative=0.
    for x,g in enumerate(groups):
     local=sorted((w for w in workers if w['head']//3==g),key=lambda w:w['diagnostic_pe'][1])
     bank=read('state',4096,y=2,x=x,w=1,h=len(local)).view(np.float32)
     ws=read('status',3,y=2,x=x,w=1,h=len(local))
     np.testing.assert_array_equal(ws[:,0],token);np.testing.assert_array_equal(ws[:,1],1)
     np.testing.assert_array_equal(ws[:,2],[w['columns'] for w in local])
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
    values=dict(packet=packet,history=history,core=core,output=output,state=actual)
    if replay:
     for name,a in values.items():np.testing.assert_array_equal(a.view(np.uint8),prior[position][name].view(np.uint8))
    elif position<2:prior[position]={n:a.copy() for n,a in values.items()}
    captures.update({f'{n}_{int(replay)}_{position}':a for n,a in values.items()});np.savez('actual.npz',**captures)
    record=dict(position=position,replay=replay,frontend_interval=True,recurrent_state_interval=True,core_interval=True,gated_interval=True,
                exact_history=True,all_return_markers=True,delayed_source_callback=True,maximum_state_error_bound_ratio=ratio,maximum_state_relative_l2=relative)
    records.append(record);progress('position_verified',**record)
  for name in ['weights','gain','parameters']:np.testing.assert_array_equal(read(name,data[name].reshape(width,-1).shape[1],bits=16),data[name].reshape(width,-1))
 result=dict(passed=True,physical=args.physical,normal_stop=True,groups=groups,workers=len(workers),positions=2 if args.smoke else 6,reset_replay_positions=0 if args.smoke else 2,
             actual_frontend_execution=True,actual_gdn_execution=True,host_injected_recurrent_results=False,full_original_value_coverage=plan['full_original_value_coverage'],
             records=records,fixture_sha256=meta['fixture_sha256'],scope=plan['scope'],full_model_speed_target_achieved=False)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
