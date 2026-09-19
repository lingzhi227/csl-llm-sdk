"""Exact compile-only admission and frozen source identity before appliance use."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILE=dict(compile_seconds=600,worker_cpu_millicores=1000,worker_memory_bytes=4<<30,
             host_address_space_bytes=4<<30,host_sampled_rss_bytes=1<<30,host_file_bytes=128<<20,
             host_log_bytes=8<<20,candidate_bytes=512<<20,host_disk_reserve_bytes=32<<30,
             application=[750,45],fabric=[762,1172],offset=[4,1],stack_bytes=4096,sram_ceiling_bytes=48128)
def verify():
    raw=(ROOT/'manifest.json').read_bytes();admission=json.loads((ROOT/'admission.json').read_bytes())
    if admission!=dict(candidate=ROOT.name,manifest_sha256=hashlib.sha256(raw).hexdigest(),profile=PROFILE,compile_only=True,automatic_retry=False):
        raise ValueError('Exact selected-program fullfit compile admission required')
    for name,spec in json.loads(raw)['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Source path')
        p=ROOT/name
        if p.is_symlink() or p.stat().st_size!=spec['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError('Frozen source changed: '+name)
    if shutil.disk_usage(ROOT).free<PROFILE['host_disk_reserve_bytes']:raise ValueError('ALCF disk reserve')
    return admission
