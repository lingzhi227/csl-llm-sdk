"""Compare actual sequential/compensated CSL RMSNorm on original activation data."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--physical',action='store_true');args=parser.parse_args()
    meta=json.loads(Path('norm-fixture.json').read_text())
    assert hashlib.sha256(Path('norm-fixture.npz').read_bytes()).hexdigest()==meta['sha256']
    with np.load('norm-fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    records=[];actual=np.zeros((len(data['hidden']),2,2880),np.uint16)
    with runtime(args.physical) as (runner,types,order):
        ids={n:runner.get_id(n) for n in ('hidden','gain','output','calls')}
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        for case in range(len(data['hidden'])):
            for name in ('hidden','gain'):
                values=np.tile(data[name][case],2).astype(np.uint32)
                runner.memcpy_h2d(ids[name],values,0,0,2,1,2880,data_type=types.MEMCPY_16BIT,**opts)
            runner.launch('normalize',nonblock=False)
            values=np.zeros(5760,np.uint32)
            runner.memcpy_d2h(values,ids['output'],0,0,2,1,2880,data_type=types.MEMCPY_16BIT,**opts)
            actual[case]=(values&65535).astype(np.uint16).reshape(2,2880)
            for method in range(2):
                raw=actual[case,method]
                a=(raw.astype(np.uint32)<<16).view(np.float32)
                assert np.isfinite(a).all()
                def ordered(x):
                    x=x.astype(np.int32);return np.where(x&32768,-(x&32767),x)
                ulp=np.abs(ordered(raw)-ordered(data['fp64'][case]))
                record=dict(case=case,method='compensated' if method else 'sequential',
                  fp64_different=int(np.count_nonzero(raw!=data['fp64'][case])),
                  torch_different=int(np.count_nonzero(raw!=data['torch'][case])),max_fp64_bf16_ulp=int(ulp.max()))
                records.append(record)
                if method==1 and not bool((ulp<=1).all()):raise ValueError('Compensated normalization exceeds one BF16 ULP from FP64')
            if case%24==23:print(json.dumps(dict(cases_completed=case+1)),flush=True)
        counts=np.zeros(2,np.uint32)
        runner.memcpy_d2h(counts,ids['calls'],0,0,2,1,1,data_type=types.MEMCPY_32BIT,**opts)
        assert np.array_equal(counts,[len(actual),len(actual)])
    np.savez('actual.npz',output=actual)
    summary={method:{key:sum(r[key] for r in records if r['method']==method)
      for key in ('fp64_different','torch_different')} for method in ('sequential','compensated')}
    assert summary['compensated']['fp64_different']<=summary['sequential']['fp64_different']
    result=dict(passed=True,physical=args.physical,normal_stop=True,cases=len(actual),summary=summary,
      scope='actual CSL RMSNorm accuracy on all-layer original and captured activations; not full-model inference',records=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
