"""Matched direct encoding and complete group quantization with retained buffers."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 direct=[];quant=[]
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['input','output','scale','ticks','calls']}
  opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,v,bits=32):
   value=np.tile(v,2);runner.memcpy_h2d(ids[name],value,0,0,2,1,v.size,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32):
   v=np.zeros(2*count,np.uint32);runner.memcpy_d2h(v,ids[name],0,0,2,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(2,count)
  def check(expected,word_input):
   got=read('output',512,16);wanted=np.full(512,65535,np.uint32);wanted[:len(expected)]=expected
   np.testing.assert_array_equal(got,np.tile(wanted,(2,1)))
   np.testing.assert_array_equal(read('input',512),np.tile(word_input,(2,1)))
   ticks=read('ticks',6,16);cycles=[]
   for row in ticks:
    t=[sum(int(row[i+k])<<(16*k) for k in range(3)) for i in (0,3)];cycles.append((t[1]-t[0])%(1<<48))
   return cycles
  for first in range(0,meta['direct_words'],512):
   n=min(512,meta['direct_words']-first);values=np.full(512,np.float32(1337).view(np.uint32),np.uint32);values[:n]=data['words'][first:first+n]
   upload('input',values);upload('output',np.full(512,65535,np.uint32),16)
   runner.launch('encode',np.uint16(n),nonblock=False);times=check(data['direct'][first:first+n],values)
   direct.append(dict(first_word=first,count=n,original_cycles=times[0],candidate_cycles=times[1]))
   if first%(512*24)==0:print(json.dumps(direct[-1]),flush=True)
  for i in range(meta['quant_groups']):
   values=np.full(512,np.float32(1337).view(np.uint32),np.uint32);values[:128]=data['groups'][i].view(np.uint32)
   upload('input',values);upload('output',np.full(512,65535,np.uint32),16)
   runner.launch('quantize',nonblock=False);times=check(data['quant'][i],values)
   scales=read('scale',1);np.testing.assert_array_equal(scales[0],scales[1])
   ulp=abs(int(scales[0,0])-int(data['scales'][i,0].view(np.uint32)));assert ulp<=1
   quant.append(dict(group=i,original_cycles=times[0],candidate_cycles=times[1],scale_ulp=ulp,bit_exact=True))
   if i%8==0:print(json.dumps(quant[-1]),flush=True)
  np.testing.assert_array_equal(read('calls',2),[[len(direct),len(quant)]]*2)
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope=meta['scope'],fixture_sha256=meta['fixture_sha256'],direct_words=meta['direct_words'],direct_batches=direct,quant_groups=quant,
  all_inputs_and_output_tails_retained=True,clock_hz=None,timing_note='Same-PE cycles for matched inputs. Group timing includes maximum, scale, division and encoding. No network or full-model token rate.')
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,direct_words=meta['direct_words'],quant_groups=len(quant))),flush=True)
if __name__=='__main__':main()
