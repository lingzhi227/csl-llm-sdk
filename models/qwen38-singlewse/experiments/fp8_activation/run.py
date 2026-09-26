"""Qualify dynamic group quantization; no CPU neural results feed the device."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');a=p.parse_args()
    meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as z:f={k:z[k] for k in z.files}
    records=[]
    with runtime(a.physical) as (runner,dtype,order):
        ids={n:runner.get_id(n) for n in ['input','codes','direct_codes','scale']}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        for i,x in enumerate(f['vectors']):
            runner.memcpy_h2d(ids['input'],x.copy(),0,0,1,1,128,data_type=dtype.MEMCPY_32BIT,**opts)
            runner.launch('compute',nonblock=False)
            q=np.zeros(128,np.uint32);d=q.copy();s=np.zeros(1,np.float32)
            for name,out in [('codes',q),('direct_codes',d),('scale',s)]:
                runner.memcpy_d2h(out,ids[name],0,0,1,1,len(out),data_type=dtype.MEMCPY_32BIT if name=='scale' else dtype.MEMCPY_16BIT,**opts)
            ulp=abs(int(s.view(np.uint32)[0])-int(f['scales'][i].view(np.uint32)[0]))
            r=dict(group=i,direct_mismatches=int(np.count_nonzero(d!=f['direct'][i])),quant_mismatches=int(np.count_nonzero(q!=f['codes'][i])),scale_ulp=ulp)
            records.append(r);Path('observations.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(r),flush=True)
            np.savez(f'actual-{i}.npz',codes=q,direct=d,scale=s)
            if r['direct_mismatches'] or r['quant_mismatches'] or ulp>1:raise ValueError('FP8 activation qualification failed')
    Path('result.json').write_text(json.dumps(dict(passed=True,normal_stop=True,physical=a.physical,full_model=False,scope=meta['scope'],groups=records),indent=2)+'\n')

if __name__=='__main__':main()
