"""Exhaustive packing and original-weight ordered-dot comparison on two PEs."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime


def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 rows=[];unpacks=[]
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','packet','scales','decoded','output','ticks','calls']}
  opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,value,bits=32):
   value=np.tile(value,2)
   runner.memcpy_h2d(ids[name],value,0,0,2,1,value.size//2,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32,kind=np.float32):
   v=np.zeros(2*count,kind);runner.memcpy_d2h(v,ids[name],0,0,2,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(2,count)
  def timing():
   raw=read('ticks',12,16,np.uint32);t=np.array([[sum(int(row[i+k])<<(16*k) for k in range(3)) for i in (0,3,6,9)] for row in raw],dtype=np.uint64)
   return [dict(total_cycles=int((int(x[3])-int(x[0]))%(1<<48)),decode_cycles=int((int(x[1])-int(x[0]))%(1<<48)),gemv_cycles=int((int(x[2])-int(x[1]))%(1<<48)),finalize_cycles=int((int(x[3])-int(x[2]))%(1<<48))) for x in t]
  for begin in range(0,65536,1024):
   values=np.arange(begin,begin+1024,dtype=np.uint32);upload('weights',values,16)
   runner.launch('unpack',nonblock=False);bits=read('decoded',2048,16,np.uint32)
   codes=np.empty(2048,np.uint32);codes[::2]=values&255;codes[1::2]=values>>8
   expected=((codes&127)<<7)|((codes&128)<<8)
   np.testing.assert_array_equal(bits,np.tile(expected,(2,1)))
   t=timing();unpacks.append(dict(first_word=begin,scalar_cycles=t[0]['total_cycles'],vector_cycles=t[1]['total_cycles']))
  for i in range(meta['cases']):
   upload('weights',data['weights'][i].astype(np.uint32),16);upload('scales',data['scales'][i]);upload('packet',data['packets'][i])
   runner.launch('compute',nonblock=False);actual=read('output',2);t=timing()
   assert np.isfinite(actual).all() and np.all(np.abs(actual.astype(np.float64)-data['exact'][i])<=data['bounds'][i])
   # Same finite nonzero FP32 bits; positive-zero accumulation may canonicalize zero signs.
   np.testing.assert_array_equal(np.where(actual[0]==0,0,actual[0]).view(np.uint32),np.where(actual[1]==0,0,actual[1]).view(np.uint32))
   assert np.array_equal(read('weights',128,16,np.uint32),np.tile(data['weights'][i],(2,1)))
   assert np.array_equal(read('packet',33,kind=np.uint32),np.tile(data['packets'][i],(2,1)))
   assert np.array_equal(read('scales',3).view(np.uint32),np.tile(data['scales'][i],(2,1)).view(np.uint32))
   rows.append(dict(case=i,scalar=t[0],vector_native=t[1],max_abs_fp64_error=float(np.max(np.abs(actual.astype(np.float64)-data['exact'][i]))),bit_exact=True))
   if i%8==0:print(json.dumps(rows[-1]),flush=True)
  assert np.array_equal(read('calls',2,kind=np.uint32),[[64,meta['cases']]]*2)
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope='Exhaustive packed-byte bit insertion and original 2x128 FP8 dot tiles; no complete projection or inference',
  packing_words_checked=65536,packing_note='All bit patterns tested as a transform; original FP8 NaN bytes are excluded from arithmetic fixtures',
  original_weight_cases=len(rows),fixture_sha256=meta['fixture_sha256'],unpack_timing=unpacks,cases=rows,all_inputs_retained=True,clock_hz=None)
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,cases=len(rows),unpack_words=65536)),flush=True)

if __name__=='__main__':main()
