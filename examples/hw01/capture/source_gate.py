"""Bind this single full-layer physical capture and its separate offline gates."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILE=dict(runtime_seconds=600,worker_cpu_millicores=2000,worker_memory_bytes=4<<30,
             host_address_space_bytes=4<<30,host_sampled_rss_bytes=1<<30,host_file_bytes=128<<20,
             host_log_bytes=8<<20,candidate_bytes=1<<30,host_disk_reserve_bytes=32<<30,
             input_count=4,copies=253,launches=9,maximum_host_transfer_bytes=16<<20,
             final_weight_storage='native_little_endian_u16_per_rectangle',
             application=[184,314],fabric=[762,1172],offset=[4,1],
             mathematical_audit_during_allocation=False,automatic_retry=False)
def verify():
    raw=(ROOT/'manifest.json').read_bytes();admission=json.loads((ROOT/'admission.json').read_bytes())
    if admission!=dict(candidate=ROOT.name,manifest_sha256=hashlib.sha256(raw).hexdigest(),profile=PROFILE,runtime_only=True,automatic_retry=False):
        raise ValueError('Exact full physical capture admission')
    for name,spec in json.loads(raw)['files'].items():
        path=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or path.stat().st_size!=spec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError('Frozen runtime source changed: '+name)
    if shutil.disk_usage(ROOT).free<PROFILE['host_disk_reserve_bytes']:raise ValueError('Host disk reserve')
    return admission
