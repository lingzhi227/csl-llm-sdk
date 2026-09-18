import hashlib,io,json
from pathlib import Path
WORK=Path(__file__).resolve().parent
PROFILE=json.loads((WORK/"accepted-profile.json").read_text())
def require(ok,message):
    if not ok: raise ValueError(message)

def load_prepared():
    import numpy as np
    folder=WORK/'prepared';raw=(folder/'preparation.json').read_bytes()
    require(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==PROFILE['preparation_receipt_sha256'],'Accepted spatial preparation identity')
    result=json.loads(raw)
    require(result['status']=='resident_source_prepared' and result['source_manifest_sha256']==PROFILE['resident_source_manifest_sha256'],'Accepted source scope')
    require(result['source_uses_CPU_or_SDK_observations'] is False and result['original_model_rerun'] is False,'Independent fixed source')
    require(result['projection_input_range']==[0,192] and result['channel_range']==result['output_range']==[0,128],'Exact spatial ranges')
    require(set(result['files'])=={'weights/gate_proj.bf16','weights/up_proj.bf16','weights/down_proj.bf16','hidden.bf16','oracle.npz'},'Five exact selected inputs')
    data={}
    for name,info in result['files'].items():
        p=folder/name;require(not p.is_symlink() and p.stat().st_size==info['bytes'] and info['bytes']<=131072,'Bounded prepared file')
        value=p.read_bytes();require(hashlib.sha256(value).hexdigest()==info['sha256'],'Exact selected source hash')
        data[name]=value
    weights={r:np.frombuffer(data['weights/'+r+'_proj.bf16'],dtype='<u2').reshape(128,128 if r=='down' else 192) for r in ('gate','up','down')}
    hidden=np.frombuffer(data['hidden.bf16'],dtype='<u2').reshape(3,192)
    require(hashlib.sha256(data['oracle.npz']).hexdigest()==PROFILE['oracle_sha256'],'Accepted oracle identity')
    with np.load(io.BytesIO(data['oracle.npz']),allow_pickle=False) as archive:
        require(set(archive.files)==set(result['arrays']) and len(archive.files)==16,'Exact source array keys')
        oracle={name:archive[name] for name in archive.files}
    for name,a in oracle.items():
        info=result['arrays'][name]
        require(a.shape==(3,128,2) and a.dtype==np.float64 and a.nbytes==6144 and hashlib.sha256(a.tobytes()).hexdigest()==info['sha256'],'Exact source endpoints')
    # No source recomputation here. The immutable accepted receipt supplies its
    # sensitivity results; driver checks never mutate or narrow these intervals.
    require(result['sensitivity']['passed'] and result['sensitivity']['all_zero_source_arrays'],'Accepted sensitivity receipt')
    return weights,hidden,oracle
