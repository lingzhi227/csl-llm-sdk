"""Freeze all original layer0 mixer projection outputs before device observations."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from weights import OriginalWeights
from reference.mixer_oracle import project_rows

started=time.monotonic();meta=json.loads(Path('tensors.json').read_text());plan=json.loads(Path('region.json').read_text())
assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
pins={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(Path('/srv/model-storage/qwen38-singlewse/model'),meta['tensors'],pins)
arrays={};cases=[];expected_shapes=[[10240,5120],[6144,5120],[48,5120],[48,5120],[5120,6144]]
assert [m['shape'] for m in plan['matrices']]==expected_shapes
for case,mode in enumerate(('mixed','zero_after','changed','replay')):
    for parts in (40,48):
        ids=np.arange(parts*128,dtype=np.uint16)
        bits=(((123+ids%5)<<7)|((ids*31+13)%128)|((ids%2)<<15)).astype(np.uint16)
        if mode=='zero_after':bits=np.where(ids%2,32768,0).astype(np.uint16)
        if mode=='changed':bits=(((117+ids%11)<<7)|((ids*17+7)%128)|(((ids+1)%2)<<15)).astype(np.uint16)
        arrays[f'input_{case}_{parts}']=bits
    reports=[]
    for mi,m in enumerate(plan['matrices']):
        name=m['tensor'];rows,columns=m['shape'];bf16=m['dtype']=='BF16'
        assert meta['tensors'][name]['shape']==m['shape'] and meta['tensors'][name]['dtype']==m['dtype']
        inputs=arrays[f'input_{case}_{columns//128}'];outputs={k:[] for k in ['bf16','ordered_fp32','fp64','bound']}
        for first in range(0,rows,64):
            count=min(64,rows-first);raw=reader.rows(name,first,count);scales=None
            if not bf16:
                scales=np.repeat(reader.rows(name+'_scale_inv',first//128,1),count,axis=0)
            result=project_rows(raw,inputs,scales)
            for key in outputs:outputs[key].append(result[key])
        for key,values in outputs.items():arrays[f'{key}_{case}_{mi}']=np.concatenate(values)
        truth=arrays[f'fp64_{case}_{mi}'];bound=arrays[f'bound_{case}_{mi}']
        ratio=float(np.linalg.norm(bound)/max(float(np.linalg.norm(truth)),1e-30))
        if mode!='zero_after' and ratio>=0.02:raise ValueError('A priori reference bound too broad')
        if mode=='replay':np.testing.assert_array_equal(arrays[f'bf16_{case}_{mi}'],arrays[f'bf16_0_{mi}'])
        reports.append(dict(tensor=name,shape=m['shape'],outputs=rows,maximum_fp64_bound=float(np.max(bound)),relative_bound_norm=ratio))
    cases.append(dict(case=case,mode=mode,matrices=reports))
with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
receipt=dict(passed=True,revision=meta['revision'],original_shards=pins,cases=cases,original_matrices=5,output_values_per_case=sum(s[0] for s in expected_shapes),
    cases_count=len(cases),fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),seconds=time.monotonic()-started,
    criterion='Original full-K ordered native FP32 and exact BF16 boundary; zero sign may be canonicalized. Independent FP64 absolute roundoff bound frozen before device output. Zero/change/replay and original bank retention required.',
    physical=False,neural_device_execution=False,full_model_speed_target_achieved=False)
Path('fixture.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
