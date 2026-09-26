"""Complete 2880-column projection: west scatter and ten eastward partial sums."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from backend import runtime

def execute(runner,types,order):
    started=time.monotonic()
    def progress(phase):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-started)),flush=True)
    meta=json.loads(Path('fixture.json').read_text())
    assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
    with np.load('fixture.npz',allow_pickle=False) as archive:f={k:archive[k] for k in archive.files}
    ids={n:runner.get_id(n) for n in ('weights','scales','input','output','calls')}
    opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
    def transfer(name,data,x,width,count,bits,read=False):
        dtype=types.MEMCPY_16BIT if bits==16 else types.MEMCPY_32BIT
        if read:runner.memcpy_d2h(data,ids[name],x,0,width,1,count,data_type=dtype,**opts)
        else:runner.memcpy_h2d(ids[name],data,x,0,width,1,count,data_type=dtype,**opts)
    weights=f['blocks'].reshape(-1).view('<u2').astype(np.uint32)
    scales=f['scales'].reshape(-1).view('<u2').astype(np.uint32)
    transfer('weights',weights,1,10,11520,16)
    transfer('scales',scales,1,10,720,16)
    progress('weights_loaded')
    records=[]
    for i,vector in enumerate(f['vectors']):
        transfer('input',vector.copy(),0,1,2880,32)
        progress(f'call_{i+1}_input_loaded')
        runner.launch('compute',nonblock=False)
        progress(f'call_{i+1}_launch_returned')
        actual=np.zeros(160,np.float32);received=np.zeros(2880,np.float32)
        counters=np.zeros(12,np.uint32)
        wb=np.zeros_like(weights);sb=np.zeros_like(scales)
        transfer('output',actual,11,1,160,32,True)
        transfer('input',received,1,10,288,32,True)
        transfer('calls',counters,0,12,1,32,True)
        progress(f'call_{i+1}_operands_and_output_read')
        transfer('weights',wb,1,10,11520,16,True)
        transfer('scales',sb,1,10,720,16,True)
        progress(f'call_{i+1}_weights_read')
        err=np.abs(actual.astype(np.float64)-f['exact'][i])
        checks=dict(numerical=bool(np.isfinite(actual).all() and np.all(err<=f['bounds'][i])),
                    scatter_exact=bool(np.array_equal(received.view(np.uint32),vector.view(np.uint32))),
                    every_pe_completed=bool(np.all(counters==i+1)),
                    weights_retained=bool(np.array_equal(wb&65535,weights) and np.array_equal(sb&65535,scales)))
        record=dict(call=i+1,checks=checks,max_abs_error=float(err.max()),max_abs_bound=float(f['bounds'][i].max()))
        records.append(record)
        np.save(f'actual-{i}.npy',actual)
        Path('observations.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if not all(checks.values()):raise ValueError('Full-width chain validation failed')
    return records

def main():
    p=argparse.ArgumentParser();p.add_argument('--physical',action='store_true');args=p.parse_args()
    with runtime(args.physical) as (runner,types,order):records=execute(runner,types,order)
    result=dict(passed=True,normal_stop=True,physical=args.physical,full_model=False,west_to_east=True,
                scope='original 160x2880 expert projection, ten resident weight PEs, two real hidden states and zero',calls=records)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
