"""Admitted native SDK entry for the fixed resident spatial qualification loop."""
import hashlib,io,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.resident_sdk_admission import PROFILE,WORK,require,verify,verify_compiled
from qwen38.wp16_affinity import current

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

def main():
    require(Path.cwd()==WORK and os.environ.get('WP16_SDK_STAGE')=='simulate','Exact admitted simulator entry')
    verify(WORK);verify_compiled(WORK);current([0])
    weights,hidden,oracle=load_prepared()
    from qwen38.resident_sdk_driver import Evidence,NativeAdapter,execute
    evidence=Evidence(WORK/'device-evidence');runner=None;executing=False
    try:
        from qwen38.resident_elf_placement import inspect_placement
        from cerebras.elf.cself import ELFMemory
        placement=inspect_placement(WORK,ELFMemory)
        with (WORK/'placement-admission.json').open('x') as stream:
            json.dump(placement,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        evidence.event('ELF_placement_checked',files=8,unique_application_coordinates=8)
        evidence.event('sdk_import_enter')
        from cerebras.sdk.runtime.sdkruntimepybind import (SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform)
        evidence.event('sdk_import_exit')
        evidence.event('construct_enter')
        runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3))
        evidence.event('construct_exit')
        backend=NativeAdapter(runner,MemcpyDataType,MemcpyOrder)
        evidence.event('symbols_resolved',symbols={name:int(value) for name,value in backend.symbols.items()})
        executing=True
        execute(backend,weights,hidden,oracle,evidence)
    finally:
        # execute owns stop once it starts. Construction/symbol failures still
        # release the runner if it exists, without hiding the original failure.
        if runner is not None and not executing:
            try:runner.stop()
            except BaseException:pass
    verify(WORK);verify_compiled(WORK,after_runtime=True);current([0])

if __name__=='__main__':main()
