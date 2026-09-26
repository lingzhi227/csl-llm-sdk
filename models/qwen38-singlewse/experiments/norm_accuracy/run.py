import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def ordered(bits):
 x=bits.astype(np.int32);return np.where(x&32768,-(x&32767),x)

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert meta['max_bf16_ulp_from_fp64']==1
 assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 records=[];actual=np.empty((65,meta['positions'],2,5120),np.uint16);started=time.monotonic()
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['input','gain','calls']};opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  for layer in range(65):
   values=np.tile(data['gain'][layer],2).astype(np.uint32)
   runner.memcpy_h2d(ids['gain'],values,0,0,2,1,5120,data_type=dtype.MEMCPY_16BIT,**opts)
   for position in range(meta['positions']):
    values=np.tile(data['inputs'][layer,position],2).astype(np.uint32)
    runner.memcpy_h2d(ids['input'],values,0,0,2,1,5120,data_type=dtype.MEMCPY_16BIT,**opts)
    runner.launch('normalize',nonblock=False);values=np.zeros(10240,np.uint32)
    runner.memcpy_d2h(values,ids['input'],0,0,2,1,5120,data_type=dtype.MEMCPY_16BIT,**opts)
    bits=(values&65535).astype(np.uint16).reshape(2,5120);actual[layer,position]=bits
    distance=np.abs(ordered(bits)-ordered(data['fp64'][layer,position]))
    assert np.isfinite((bits.astype(np.uint32)<<16).view(np.float32)).all()
    r=dict(layer=layer,position=position,baseline_different=int(np.count_nonzero(distance[0])),compensated_different=int(np.count_nonzero(distance[1])),
      baseline_max_ulp=int(distance[0].max()),compensated_max_ulp=int(distance[1].max()))
    records.append(r);assert r['compensated_max_ulp']<=1,r
   print(json.dumps(dict(layers_completed=layer+1,seconds=time.monotonic()-started)),flush=True)
  counts=np.zeros(2,np.uint32);runner.memcpy_d2h(counts,ids['calls'],0,0,2,1,1,data_type=dtype.MEMCPY_32BIT,**opts)
  assert np.array_equal(counts,[meta['cases']]*2)
  gains=np.zeros(10240,np.uint32);runner.memcpy_d2h(gains,ids['gain'],0,0,2,1,5120,data_type=dtype.MEMCPY_16BIT,**opts)
  assert np.array_equal(gains&65535,np.tile(data['gain'][-1],2))
 np.savez('actual.npz',outputs=actual)
 before=sum(r['baseline_different'] for r in records);after=sum(r['compensated_different'] for r in records);assert after<=before
 result=dict(passed=True,physical=a.physical,normal_stop=True,full_model=False,cases=len(records),values_per_method=len(records)*5120,
  baseline_different=before,compensated_different=after,max_compensated_bf16_ulp=max(r['compensated_max_ulp'] for r in records),records=records,
  scope=meta['scope'],fixture_sha256=meta['fixture_sha256'])
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
