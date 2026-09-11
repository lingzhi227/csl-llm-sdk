"""Unexecuted portable source configuration template.

The numerical helper policies are unchanged. Private cache/Python locations are
placeholders; accepted source preparation has its own recorded frozen hash.
No source preparation is executed by importing this public template. Configure,
freeze and admit a separate source candidate before any new original-data read.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

CANDIDATE = 'wp16-connected-source-001'
BASE = Path('/path/to/qwen38-cache')
WORK = BASE / CANDIDATE
PARTIAL = 'wp16-partial-source-001/prepared/'
PREFIX = 'wp16-prefix-source-001/prepared/'
INPUTS = {
    PARTIAL+'weights/gate_proj.bf16': dict(bytes=1310720,sha256='e2adddcd0ae3eae6910e6897709c5b9063313fab45586b740e13c8fb1204d2ad'),
    PARTIAL+'weights/up_proj.bf16': dict(bytes=1310720,sha256='7d4dc6bbd324bfea4143f2647d17789a805451e872b8b2545af665911f61a378'),
    PARTIAL+'weights/down_proj.bf16': dict(bytes=32768,sha256='f7604a9da23ed57eceb60571dff046ade5e0e4bf0a701a4e02fe47459343130b'),
    PARTIAL+'hidden.bf16': dict(bytes=40960,sha256='cc11620f3873f982c0f3424d00f5992b0fa03241b6df3b259f9b9cbf9a0b0905'),
    PREFIX+'prefixes.npz': dict(bytes=189406,sha256='ffb93bf9923cff46005c4fbc2549778ce8ebe6a30417d17783c31a7f488ee733'),
    PREFIX+'preparation.json': dict(bytes=3371,sha256='ac2b5e7e2989795d02be5d05a6b482e3eafa6981a933b0ddade8d1c9d4a1708e'),
}
HELPERS = {
    'mlp_streamed_source.py':'e95ac4b2cd8168b5789dd5b42ceefe8ef8ecf487d1ef2c77ecb8608e4ec5f420',
    'mlp_numerics.py':'1ea55329b6623cda7d0831fc712578dc0289d642f35f51436817531988e9406c',
    'mlp_reference_checks.py':'467a2dec4b1a3273e7875e2fb119b3c9e06b148b11f98c8544ea7b30784d6859',
}
THREAD_ENV = dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',
    PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
LIMITS = {'memory.max':'268435456','memory.swap.max':'0','pids.max':'64'}
PROFILE = dict(package='WP16',candidate=CANDIDATE,candidate_limit=1,
    scope='connected_four_PE_three_generation112_column_source_only',input_root=str(BASE),inputs=INPUTS,
    helper_sha256=HELPERS,input_count=6,input_bytes=2887945,input_read_once=True,maximum_read_chunk_bytes=1048576,
    original_input_indices=[0,1,3],generation_input_contraction=[[1,0,1],[2,1,2],[3,3,3]],
    memory_bytes=268435456,swap_bytes=0,cpu_affinity=[0],worker_threads=1,tasks_max=64,
    cpu_limit_method='single_logical_CPU_affinity_not_exclusive_or_bandwidth_quota',
    planned_seconds=10,seconds=30,file_bytes=524288,core_bytes=0,
    source_bytes=1048576,ssd_run_bytes=4194304,prepared_bytes=1048576,
    archive_bytes=131072,phase_journal_bytes=32768,phase_journal_records=64,log_bytes=131072,
    ram_reserve_bytes=8589934592,ssd_reserve_bytes=34359738368,ssd_cache_bytes=21474836480,
    projection_shape=[128,112],down_shape=[128,128],source_arrays=11,source_array_shape=[3,128,2],numeric_bytes=67584,
    selected_weight_bytes=90112,selected_hidden_bytes=672,linear_bounds_calls=24,
    cumulative_weight_elements=159744,maximum_FP64_rows=64,maximum_FP64_columns=128,
    scalar_silu_calls=384,scalar_product_calls=384,required_dense_anchor_endpoints=512,
    require_disjoint_BF16_stages=['gate_bf16','up_bf16','product_bf16','down_bf16'],
    require_changed_down_excludes_zero=True,require_all_zero_source_stages=True,
    HDD_reads=False,model_rerun=False,Torch=False,SDK=False,GPU=False,network=False,
    automatic_retry=False,source_uses_CPU_or_SDK_observations=False)
STEPS = dict(steps=[dict(name='connected_source',seconds=30,
    argv=['/path/to/qualified-source-python','prepare_source.py'])])

def require(condition,message):
    if not condition: raise ValueError(message)

def exact(a,b): return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)

def safe_path(root,name):
    relative=Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative)==name,'Safe connected source path')
    path=root/relative
    require(not any(p.is_symlink() for p in (path,*path.parents)),'No source path symlinks')
    return path

def read_selected(root,name,expected,ranges):
    """Hash a file in one sequential pass; retain only ordered exact byte ranges."""
    require(type(expected['bytes']) is int and 0 < expected['bytes'] <= 2097152,'Bounded source input')
    require(bool(ranges) and all(type(a) is int and type(b) is int and 0<=a<b<=expected['bytes'] for a,b in ranges),'Valid selection ranges')
    require(all(ranges[i-1][1]<=ranges[i][0] for i in range(1,len(ranges))),'Ordered disjoint selection ranges')
    retained=sum(b-a for a,b in ranges)
    require(retained<=262144 and len(ranges)<=128,'Bounded selected bytes/ranges')
    path=safe_path(root,name); h=hashlib.sha256(); selected=bytearray(); offset=0
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size==expected['bytes'],'Exact regular source input length')
        while offset<before.st_size:
            raw=stream.read(min(1048576,before.st_size-offset))
            require(bool(raw),'Unexpected short source read')
            h.update(raw); end=offset+len(raw)
            for first,last in ranges:
                lo,hi=max(first,offset),min(last,end)
                if lo<hi: selected.extend(raw[lo-offset:hi-offset])
            offset=end
        after=os.fstat(stream.fileno())
    require(all(getattr(before,k)==getattr(after,k) for k in
        ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'Source input changed during read')
    require(offset==expected['bytes'] and h.hexdigest()==expected['sha256'],'Pinned source input hash')
    require(len(selected)==retained,'Exact selected byte count')
    return bytes(selected)

def admission(manifest_sha256,authorized=True):
    return dict(package='WP16',candidate=CANDIDATE,candidate_limit=1,
        connected_source_generation_authorized=authorized,manifest_sha256=manifest_sha256,
        profile=PROFILE,steps=STEPS,model_rerun_authorized=False,
        Torch_authorized=False,SDK_authorized=False,network_authorized=False,HDD_reads_authorized=False)

def verify(work=WORK):
    require(work==WORK,'Exact connected source working directory')
    path=safe_path(work,'source-manifest.json')
    require(path.is_file() and path.stat().st_size<=65536,'Bounded connected manifest')
    raw=path.read_bytes(); digest=hashlib.sha256(raw).hexdigest(); manifest=json.loads(raw)
    require(manifest.get('package')=='WP16' and manifest.get('candidate')==CANDIDATE,'Connected source manifest identity')
    require(15<=len(manifest['files'])<=48,'Bounded connected file inventory')
    total=0
    for name,expected in manifest['files'].items():
        path=safe_path(work,name); info=path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_size<=524288,'Small regular connected source')
        total+=info.st_size
        require(hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Frozen connected source changed: '+name)
    require(total<=PROFILE['source_bytes'],'Connected source byte budget')
    for name,expected in HELPERS.items():
        require(manifest['files'].get('core/qwen38/'+name)==expected,'Accepted source math unchanged')
    for name,expected in [('profile.json',PROFILE),('steps.json',STEPS),('controller-admission.json',admission(digest))]:
        path=safe_path(work,name)
        require(path.is_file() and path.stat().st_size<=65536,'Bounded connected admission metadata')
        require(exact(json.loads(path.read_bytes()),expected),'Exact connected admission/profile/steps required')
    return digest
