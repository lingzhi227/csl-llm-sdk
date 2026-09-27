"""Resident original-weight slots plus unchanged regional arithmetic/transport."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def main():
 p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
 meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
 with np.load('fixture.npz',allow_pickle=False) as f:data={n:f[n] for n in f.files}
 w,h=meta['application'];caps=np.array(json.loads(Path('region.json').read_text())['bank_counts']);assert caps.shape==(h,w)
 rows=[];retained={}
 with runtime(a.physical) as (runner,dtype,order):
  ids={n:runner.get_id(n) for n in ['weights','scale','bf16_reserve','metadata','slot','raw','packet','output','row_result','ticks','audit']}
  opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
  def upload(name,v,bits=32,x=0,y=0,ww=None,hh=None):
   ww=w if ww is None else ww;hh=h if hh is None else hh;v=np.ascontiguousarray(v)
   runner.memcpy_h2d(ids[name],v.reshape(-1),x,y,ww,hh,v.size//(ww*hh),data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
  def read(name,count,bits=32,kind=np.float32,x=0,y=0,ww=None,hh=None):
   ww=w if ww is None else ww;hh=h if hh is None else hh;v=np.zeros(ww*hh*count,kind)
   runner.memcpy_d2h(v,ids[name],x,y,ww,hh,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return v.reshape(hh,ww,count)
  def bits(v):return np.where(v==0,np.float32(0),v).astype(np.float32).view(np.uint32)
  for y in range(h):
   for x in range(w):
    cap=int(caps[y,x]);mapping=(cap-1-np.arange(cap))%4
    values={'weights':data['weights'][mapping,y,x].astype(np.uint32).reshape(-1),'scale':data['scales'][mapping,y,x,0],
      'bf16_reserve':(np.arange(3072,dtype=np.uint32)*13+7+x+17*y)&65535,'metadata':np.arange(cap+12,dtype=np.uint32)*65537+19+x+17*y}
    retained[y,x]=values
    for name,v in values.items():upload(name,v,16 if name in ['weights','bf16_reserve'] else 32,x,y,1,1)
  expected_audit=np.zeros((h,w,8),np.uint32);expected_audit[:,:,:4]=1;expected_audit[0,:,4]=1;expected_audit[:,0,5]=1;expected_audit[:-1,0,6]=1;expected_audit[1:,0,7]=1
  for band in ['first','middle','last']:
   for case in range(4):
    residue=(caps-1-case)%4
    selected=residue if band=='first' else residue+4*((caps//2-residue)//4) if band=='middle' else caps-1-case
    assert np.all((caps-1-selected)%4==case) and np.all((selected>=0)&(selected<caps))
    upload('slot',selected.astype(np.uint32),16);upload('raw',data['raw'][case]);pair={}
    for mode in (0,1):
     started=time.perf_counter();runner.launch('execute',np.uint16(mode),nonblock=False)
     actual=read('output',2);result=read('row_result',2);audit=read('audit',8,kind=np.uint32);host=time.perf_counter()-started
     np.testing.assert_array_equal(audit,expected_audit);np.testing.assert_array_equal(bits(actual),bits(data['ordered'][case]))
     assert np.isfinite(actual).all() and np.all(np.abs(actual.astype(np.float64)-data['exact'][case])<=data['bounds'][case])
     np.testing.assert_array_equal(bits(result[:,0]),bits(actual[:,-1]))
     np.testing.assert_array_equal(read('packet',65,kind=np.uint32),data['wire'][case])
     np.testing.assert_array_equal(read('slot',1,16,np.uint32)[:,:,0],selected)
     np.testing.assert_array_equal(read('raw',33,kind=np.uint32),data['raw'][case])
     ticks=read('ticks',6,16,np.uint32)[0,0];start=sum(int(ticks[k])<<(16*k) for k in range(3));end=sum(int(ticks[3+k])<<(16*k) for k in range(3))
     row=dict(band=band,case=case,overlapped=bool(mode),root_region_cycles=(end-start)%(1<<48),host_launch_through_result_audit_seconds=host,selected_slots=selected.tolist(),bit_exact=True)
     rows.append(row);pair[mode]=actual.copy();print(json.dumps(row),flush=True)
    np.testing.assert_array_equal(bits(pair[0]),bits(pair[1]))
  for (y,x),values in retained.items():
   for name,v in values.items():
    size=16 if name in ['weights','bf16_reserve'] else 32;kind=np.float32 if name=='scale' else np.uint32
    np.testing.assert_array_equal(read(name,v.size,size,kind,x,y,1,1).reshape(-1),v)
 result=dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope='Combined resident original-weight FP8 banks and P5 regional communication; variable role capacities; BF16/metadata arrays retained sentinels, no BF16 execution or full-model scheduler',
  application=[w,h],bank_counts=caps.tolist(),fixture_sha256=meta['fixture_sha256'],cases=rows,all_storage_retained=True,clock_hz=None,
  timing_note='Source encoding, weight decode, multicast, local dot, ordered reduction and root completion; same root timestamp, host launch-entry skew included. Not full-model timing.')
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,physical=a.physical,epochs=len(rows))),flush=True)
if __name__=='__main__':main()
