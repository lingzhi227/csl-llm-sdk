"""Equal-work local timing only, with original resident banks and exact gates."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
 records=[];captured={};width=meta['application'][0];assert width in [4,8]
 scaling=meta.get('compare_scaling',False);input_loop=meta.get('compare_input_loop');pairing=bool(input_loop)
 paired=scaling or pairing;assert paired==(width==8) and not(scaling and pairing)
 def canon(value):return np.where(value==0,np.float32(0),value).astype(np.float32).view(np.uint32)
 with runtime(a.physical) as (runner,dtype,order):
  ids={name:runner.get_id(name) for name in ['bank','packet','output','ticks','audit']}
  opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,value):
   value=np.ascontiguousarray(value,dtype=np.uint32);runner.memcpy_h2d(ids[name],value.reshape(-1),0,0,width,1,value.shape[-1],data_type=dtype.MEMCPY_32BIT,**opts)
  def read(name,n,bits=32):
   value=np.zeros((width,n),np.uint32);runner.memcpy_d2h(value.reshape(-1),ids[name],0,0,width,1,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return value
  upload('bank',data['bank']);np.testing.assert_array_equal(read('audit',4),0)
  for case in meta['physical_cases'] if a.physical else meta['simulation_cases']:
   desc=meta['cases'][case];upload('packet',data['packet'][case])
   for decode_each in desc['decode_modes']:
    runner.launch('measure',np.uint16(desc['kind']),np.uint16(decode_each),nonblock=False)
    audit=read('audit',4);np.testing.assert_array_equal(audit,np.tile([32,desc['kind'],decode_each,256],(width,1)))
    actual=read('output',16).view(np.float32);np.testing.assert_array_equal(canon(actual),canon(data['ordered'][case]))
    error=np.abs(actual.astype(np.float64)-data['truth'][case]);assert np.isfinite(actual).all() and np.all(error<=data['bounds'][case])
    if paired:np.testing.assert_array_equal(canon(actual[:4]),canon(actual[4:]))
    np.testing.assert_array_equal(read('packet',65),data['packet'][case])
    ticks=read('ticks',6,bits=16);elapsed=[]
    for halves in ticks:
     cycles=(sum(int(halves[j+3])<<(16*j) for j in range(3))-sum(int(halves[j])<<(16*j) for j in range(3)))%(1<<48)
     assert 0<cycles<10000000;elapsed.append(cycles)
    record=dict(desc,decode_each=decode_each,repetitions=32,total_cycles_by_shape=elapsed,
                cycles_per_invocation_by_shape=[v/32 for v in elapsed],max_fp64_error=float(error.max()),
                matched_scalar_to_vector_ratios=[elapsed[i]/elapsed[i+4] for i in range(4)] if scaling else None,
                matched_single_to_candidate_ratios=[elapsed[i]/elapsed[i+4] for i in range(4)] if pairing else None)
    records.append(record);captured['output_%d_%d'%(case,decode_each)]=actual;captured['ticks_%d_%d'%(case,decode_each)]=ticks
    print(json.dumps(record),flush=True)
  np.testing.assert_array_equal(read('bank',8814),data['bank']);np.savez('actual.npz',**captured)
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,application=[width,1],scope=meta['scope'],
             fixture_sha256=meta['fixture_sha256'],shapes=meta['shapes'],cases=records,all_weights_retained=True,
             all_packets_and_counters_exact=True,all_ordered_outputs_exact=True,matched_scaling_outputs_exact=scaling,matched_input_loop_outputs_exact=pairing,compare_input_loop=input_loop,
             timing_note='EachPE measures32 local invocations, including loop/control and callback-free counter overhead. Mode0 excludes one-time FP8 decode; mode1 includes decode in every FP8 invocation. No communication, reduction, complete matrix or dependent-model timing. Different shape inputs cover different original slices; equal256MAC count only.',
             clock_hz=None,full_model_speed_target_achieved=False)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical)),flush=True)

if __name__=='__main__':main()
