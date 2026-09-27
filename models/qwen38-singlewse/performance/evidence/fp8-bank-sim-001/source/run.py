"""Maximum-occupancy retained bank and synchronous predecode ownership qualification.

Identical original FP8 fixtures fill112 slots. BF16 and metadata capacity buffers
contain non-model sentinel patterns. No full-model packing, BF16 computation,
network receive or asynchronous overlap is claimed by this bounded probe.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 indexes=np.arange(112)%meta['cases'];weights=data['weights'][indexes].astype(np.uint32).reshape(-1)
 scales=data['scales'][indexes,0];bf16=(np.arange(3072,dtype=np.uint32)*13+7)&65535
 metadata=np.arange(124,dtype=np.uint32)*65537+19
 rows=[]
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','bf16_reserve','metadata','packet','scales','output','ticks','calls']}
  opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,value,bits=32):
   v=np.tile(value,2);runner.memcpy_h2d(ids[name],v,0,0,2,1,value.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32,kind=np.float32):
   v=np.zeros(2*count,kind);runner.memcpy_d2h(v,ids[name],0,0,2,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(2,count)
  upload('weights',weights,16);upload('scales',scales);upload('bf16_reserve',bf16,16);upload('metadata',metadata)
  # Every local slot including both boundaries; alternating endpoints exercises reuse.
  order_slots=[i if i%2==0 else 112-i for i in range(112)]
  assert sorted(order_slots)==list(range(112))
  for ordinal,slot in enumerate(order_slots):
   i=int(indexes[slot]);raw=data['packets'][i,:32].copy().view(np.uint8).astype(np.uint16)
   encoded=(((raw&127)<<7)|((raw&128)<<8)).astype('<u2')
   packet=np.zeros(130,np.uint32);packet[:64]=encoded.view('<u4');packet[64]=data['packets'][i,32]
   packet[65:]=np.arange(65,dtype=np.uint32)+ordinal+1
   upload('packet',packet)
   runner.launch('prefetch',np.uint16(slot),nonblock=False)
   runner.launch('compute',np.uint16(slot),nonblock=False)
   actual=read('output',2);assert np.isfinite(actual).all()
   assert np.all(np.abs(actual.astype(np.float64)-data['exact'][i])<=data['bounds'][i])
   np.testing.assert_array_equal(actual[0].view(np.uint32),actual[1].view(np.uint32))
   ticks=read('ticks',12,16,np.uint32)
   times=[]
   for row in ticks:
    t=[sum(int(row[j+k])<<(16*k) for k in range(3)) for j in (0,3,6,9)]
    times.append(dict(prefetch_cycles=(t[1]-t[0])%(1<<48),compute_cycles=(t[3]-t[2])%(1<<48),sum_cycles=((t[1]-t[0])+(t[3]-t[2]))%(1<<48)))
   assert np.array_equal(read('packet',130,kind=np.uint32),np.tile(packet,(2,1)))
   rows.append(dict(slot=slot,fixture_case=i,inline=times[0],prefetched=times[1],bit_exact=True,max_abs_fp64_error=float(np.max(np.abs(actual.astype(np.float64)-data['exact'][i])))))
   if ordinal%28==0:print(json.dumps(rows[-1]),flush=True)
  for name,v,bits in [('weights',weights,16),('scales',scales,32),('bf16_reserve',bf16,16),('metadata',metadata,32)]:
   actual=read(name,v.size,bits,np.float32 if name=='scales' else np.uint32)
   np.testing.assert_array_equal(actual,np.tile(v,(2,1)))
  np.testing.assert_array_equal(read('calls',2,kind=np.uint32),[[112,112]]*2)
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope='Maximum-occupancy112 FP8 and12 reserved BF16 bank storage; all FP8 slots and synchronous predecode ownership; no network overlap or full scheduler',
  fixture_sha256=meta['fixture_sha256'],all_storage_retained=True,cases=rows,clock_hz=None,
  timing_note='Prefetch and compute measured separately; their sum is reported. Prepared operand encoding is a test fixture, not a qualified on-device producer. No time is claimed hidden.')
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,cases=len(rows))),flush=True)
if __name__=='__main__':main()
