"""Initialize once, two autonomous96-epoch sequences, then full retention."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 w,h=meta['application'];ww,wh=meta['workers'];cx,cy=meta['controller'];owners=meta['owners'];rows=[]
 document=json.loads(Path('region.json').read_text())
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','scale','bfweights','request','expected','failure','local_result','aggregate','audit','requests','results','ticks']};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,v,x=0,y=0,bits=32):
   v=np.ascontiguousarray(v);hh,ww,count=v.shape;runner.memcpy_h2d(ids[name],v.reshape(-1),x,y,ww,hh,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,x=0,y=0,width=ww,height=wh,bits=32):
   v=np.zeros(height*width*count,np.uint32);runner.memcpy_d2h(v,ids[name],x,y,width,height,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(height,width,count)
  def canon(v):return np.where(v==0,np.float32(0),v).astype(np.float32).view(np.uint32)
  print(json.dumps(dict(phase='initialize_banks_and_request_oracle')),flush=True);started=time.perf_counter()
  upload('weights',data['weights'].astype(np.uint32),bits=16);upload('scale',data['scales']);upload('bfweights',data['bfweights'].astype(np.uint32),bits=16);upload('expected',data['expected']);upload('requests',data['requests'].reshape(1,1,-1),cx,cy);initialization=time.perf_counter()-started
  for run,spec in enumerate(meta['runs']):
   rounds,first=spec['rounds'],spec['first'];sequence=[(first+k)%12 for k in range(rounds)];print(json.dumps(dict(phase='arm_run',run=run,**spec)),flush=True)
   started=time.perf_counter();runner.launch('initialize',np.uint16(rounds),np.uint16(first),nonblock=False);armed=read('audit',8,width=w,height=h,bits=16);wanted=np.zeros_like(armed);wanted[:,:,0]=1;np.testing.assert_array_equal(armed,wanted);arm=time.perf_counter()-started
   print(json.dumps(dict(phase='autonomous_start',run=run)),flush=True);started=time.perf_counter();runner.launch('start',nonblock=False)
   result=read('results',rounds*4,cx,cy,1,1).reshape(rounds,4);host=time.perf_counter()-started
   # One terminal audit, after the loop. Root arrival may precede last sender callback.
   started=time.perf_counter();deadline=started+5;terminal_polls=0
   while True:
    audit=read('audit',8,width=w,height=h,bits=16);terminal_polls+=1
    if np.all(audit[:,:,6]==1):break
    if time.perf_counter()>deadline:raise AssertionError('Terminal callback audit timeout: '+str(audit.tolist()))
   tail=time.perf_counter()-started;failure=read('failure',4);assert not np.any(failure),failure.tolist()
   expected=np.zeros_like(audit);expected[:,:,[0,6]]=1
   for rank,(x,y) in enumerate(owners):expected[y,x,1]=rounds;expected[y,x,2]=rounds;expected[y,x,3]=rounds*len(document['trees'][0][rank]['children']);expected[y,x,4]=rounds;expected[y,x,5]=rounds
   expected[cy,cx,1]=rounds;expected[cy,cx,2]=rounds;np.testing.assert_array_equal(audit,expected)
   x,y=owners[0];actual=np.ascontiguousarray(result[:,:2]).view(np.float32);np.testing.assert_array_equal(canon(actual),canon(data['tree'][sequence,y,x]));np.testing.assert_array_equal(result[:,2],np.full(rounds,6,np.uint32));np.testing.assert_array_equal(result[:,3],np.arange(rounds,dtype=np.uint32))
   error=np.abs(actual.astype(np.float64)-data['exact'][sequence,y,x]);assert np.isfinite(actual).all() and np.all(error<=data['bounds'][sequence,y,x])
   last=sequence[-1];np.testing.assert_array_equal(canon(read('local_result',2).view(np.float32)),canon(data['local'][last]));aggregate=read('aggregate',4);np.testing.assert_array_equal(canon(np.ascontiguousarray(aggregate[:,:,:2]).view(np.float32)),canon(data['tree'][last]));np.testing.assert_array_equal(aggregate[:,:,2],data['count'][last]);np.testing.assert_array_equal(aggregate[:,:,3],np.full((wh,ww),rounds-1,np.uint32));np.testing.assert_array_equal(read('request',66),np.broadcast_to(data['requests'][last],(wh,ww,66)))
   ticks=read('ticks',rounds*6,cx,cy,1,1,bits=16).reshape(rounds,6);starts=[sum(int(t[k])<<(16*k) for k in range(3)) for t in ticks];ends=[sum(int(t[3+k])<<(16*k) for k in range(3)) for t in ticks]
   cycles=[(b-a)%(1<<48) for a,b in zip(starts,ends)];gaps=[(starts[k+1]-ends[k])%(1<<48) for k in range(rounds-1)];total=(ends[-1]-starts[0])%(1<<48);assert total==sum(cycles)+sum(gaps)
   record=dict(run=run,**spec,sequence=sequence,request_to_result_cycles=cycles,inter_epoch_cycles=gaps,autonomous_total_cycles=total,cycles_per_epoch=total/rounds,max_abs_root_fp64_error=float(error.max()),host_start_through_root_history_seconds=host,host_arm_and_ready_seconds=arm,terminal_audit_seconds=tail,terminal_audit_polls=terminal_polls);rows.append(record);print(json.dumps({k:v for k,v in record.items() if not isinstance(v,list)}),flush=True)
  print(json.dumps(dict(phase='full_retention')),flush=True)
  np.testing.assert_array_equal(read('weights',112*128,bits=16),data['weights']);np.testing.assert_array_equal(read('scale',112),data['scales'].view(np.uint32));np.testing.assert_array_equal(read('bfweights',12*256,bits=16),data['bfweights']);np.testing.assert_array_equal(read('expected',48),data['expected'].view(np.uint32));np.testing.assert_array_equal(read('requests',12*66,cx,cy,1,1).reshape(12,66),data['requests'])
 r=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope=meta['scope'],application=[w,h],fixture_sha256=meta['fixture_sha256'],runs=rows,host_initialization_seconds=initialization,all_banks_oracles_requests_retained=True,weight_uploads=1,clock_hz=None,
  timing_note='Same-controller request issue to tagged numerical root result, including multicast/dispatch/arithmetic/reduction/return. Total includes actual controller gaps between96 epochs. Initial host readiness and final callback audit separate; no per-epoch host access. Last sender callbacks may trail final root result and are all verified. Independent preloaded neural inputs, no token throughput or assumed clock.')
 Path('result.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,epochs=192)),flush=True)
if __name__=='__main__':main()
