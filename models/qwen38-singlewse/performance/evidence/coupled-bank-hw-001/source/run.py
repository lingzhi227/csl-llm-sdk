"""Complete two-stage window checks, typed dots and original-bank readback."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
 workers=meta['workers'];document=json.loads(Path('region.json').read_text());chosen=meta['physical_cases'] if a.physical else meta['simulation_cases'];records=[];captures={}
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['bank','control','packet','local_result','audit','ticks','aggregate','left','right','probe']};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,value,x,y,w,h,bits=32):
   value=np.ascontiguousarray(value);n=value.size//(w*h);runner.memcpy_h2d(ids[name],value.reshape(-1),x,y,w,h,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,n,x=0,y=0,w=4,h=5,bits=32):
   value=np.zeros(w*h*n,np.uint32);runner.memcpy_d2h(value,ids[name],x,y,w,h,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return value.reshape(h,w,n)
  def canon(v):return np.where(v==0,np.float32(0),v).astype(np.float32).view(np.uint32)
  print(json.dumps(dict(phase='exhaustive_shift_decoder')),flush=True);runner.launch('test_unpack',nonblock=False)
  probe=read('probe',4);wanted=np.zeros((5,4),np.uint32);wanted[1:,0]=16384
  np.testing.assert_array_equal(probe[...,0],wanted);assert not np.any(probe[...,1])
  assert np.all(probe[1:,0,2:] > 0)
  decoder=dict(packed_patterns=int(probe[...,0].sum()),mismatches=int(probe[...,1].sum()),reference_decode_cycles=probe[1:,0,2].tolist(),shift_decode_cycles=probe[1:,0,3].tolist())
  print(json.dumps(decoder),flush=True)
  start=time.perf_counter();print(json.dumps(dict(phase='upload_resident_banks')),flush=True)
  packed_bank=np.zeros((4,3,8814),np.uint32)
  for d in workers:
   r=d['rank'];fp=d['fp8_slots'];x=d['x']-1;y=d['y']-1
   packed_bank[y,x,:fp*64]=data['weights_'+str(r)]
   packed_bank[y,x,fp*64:fp*65]=data['scale_'+str(r)].view(np.uint32)
   b=data['bfweights_'+str(r)];packed_bank[y,x,fp*65:fp*65+len(b)]=b
  upload('bank',packed_bank,1,1,3,4)
  initialization=time.perf_counter()-start
  expected=np.zeros((5,4,14),np.uint32);expected[...,0]=1;expected[...,7]=1
  expected[0,0,[3,4,5]]=1;expected[0,1:,[1,3,4,5,6]]=1
  expected[1:,1:,1:3]=1;expected[1:,1:,6]=1;expected[1:,1:,11:13]=1;expected[0,0,13]=1
  for node in document['tree']:
   x,y=node['xy'];expected[y,x,9]=int(len(node['children'])>0);expected[y,x,10]=int(len(node['children'])>1)
  for case in chosen:
   print(json.dumps(dict(phase='filtered_epoch',case=case)),flush=True);start=time.perf_counter()
   upload('control',data['control'][case].astype(np.uint32),0,0,4,5,16);period=meta['cases'][case]['period_words']
   upload('packet',data['stream'][case,:period],0,0,1,1);runner.launch('arm',nonblock=False)
   arming=time.perf_counter()-start;start=time.perf_counter();runner.launch('start',nonblock=False);host=time.perf_counter()-start
   observed=read('audit',14,bits=16)
   if not np.array_equal(observed,expected):
    partial=read('aggregate',9);left=read('left',3);right=read('right',3)
    diagnostic=dict(case=case,pe_counters=[dict(x=n['xy'][0],y=n['xy'][1],rank=n['rank'],subtree=n['size'],audit=observed[n['xy'][1],n['xy'][0]].tolist(),child_and_join_counts=[int(left[n['xy'][1],n['xy'][0],2]),int(right[n['xy'][1],n['xy'][0],2]),int(partial[n['xy'][1],n['xy'][0],8])]) for n in document['tree']])
    Path('diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n');print(json.dumps(diagnostic),flush=True)
   np.testing.assert_array_equal(observed,expected)
   packet=read('packet',65,1,1,3,4);np.testing.assert_array_equal(packet,data['packets'][case])
   relay=read('packet',130,1,0,3,1)
   for x,offset in enumerate(meta['cases'][case]['pair_offsets_words']):np.testing.assert_array_equal(relay[0,x],data['stream'][case,offset:offset+130])
   actual=read('local_result',2,1,1,3,4).view(np.float32);np.testing.assert_array_equal(canon(actual),canon(data['local'][case]))
   err=np.abs(actual.astype(np.float64)-data['truth'][case]);assert np.isfinite(actual).all() and np.all(err<=data['bounds'][case])
   aggregate=read('aggregate',9)[...,6:9];tree_value=np.ascontiguousarray(aggregate[1:,1:,:2]).view(np.float32)
   np.testing.assert_array_equal(canon(tree_value),canon(data['trees'][case]));np.testing.assert_array_equal(aggregate[1:,1:,2],data['counts'])
   terr=np.abs(tree_value.astype(np.float64)-data['tree_truth'][case]);assert np.all(terr<=data['tree_bounds'][case])
   np.testing.assert_array_equal(aggregate[0,0],aggregate[1,1]);assert aggregate[0,0,2]==12
   ticks=read('ticks',6,0,0,1,1,bits=16).reshape(6)
   cycles=(sum(int(ticks[3+j])<<(16*j) for j in range(3))-sum(int(ticks[j])<<(16*j) for j in range(3)))%(1<<48)
   assert 0<cycles<10000000
   captures['result_'+str(case)]=actual.copy();captures['packets_'+str(case)]=packet;captures['ticks_'+str(case)]=ticks;captures['aggregate_'+str(case)]=aggregate
   source=meta['cases'][case]['replay_of']
   if source is not None:
    np.testing.assert_array_equal(canon(actual),canon(captures['result_'+str(source)]));np.testing.assert_array_equal(aggregate,captures['aggregate_'+str(source)])
   record=dict(meta['cases'][case],max_abs_fp64_error=float(err.max()),source_send_to_result_return_cycles=cycles,max_tree_abs_fp64_error=float(terr.max()),host_upload_and_arm_seconds=arming,host_start_seconds=host)
   records.append(record);print(json.dumps(record),flush=True)
  print(json.dumps(dict(phase='full_bank_retention')),flush=True)
  np.testing.assert_array_equal(read('bank',8814,1,1,3,4),packed_bank)
  np.savez('actual.npz',**captures)
 result=dict(passed=True,normal_stop=True,physical=a.physical,application=[4,5],full_model=False,scope=meta['scope'],
  fixture_sha256=meta['fixture_sha256'],cases=records,all_weights_retained=True,original_weights=True,weight_uploads=1,bank_transfer_bytes=423072,retained_padding_bytes=1512,
  counter_windows_exact=True,all_packet_and_teardown_counters_exact=True,all_subtrees_exact=True,controller_result_exact=True,exhaustive_decoder=decoder,host_initialization_seconds=initialization,
  timing_note='Same source-controller timestamp from horizontal send to full twelve-participant result return. Includes two-stage input, native dots, ordered tree and return; excludes host upload/arm, final send callbacks and retention. Independent operands and partial K; no full-matrix/model rate claim.',clock_hz=None,full_model_speed_target_achieved=False)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical)),flush=True)
if __name__=='__main__':main()
