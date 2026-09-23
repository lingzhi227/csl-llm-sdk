"""Exact complete Layer3 compile-only admission; no runtime authorization."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILE={'compile_seconds': 600, 'worker_cpu_millicores': 1000, 'worker_memory_bytes': 8589934592, 'host_address_space_bytes': 4294967296, 'host_sampled_rss_bytes': 1073741824, 'host_file_bytes': 134217728, 'host_log_bytes': 8388608, 'candidate_bytes': 536870912, 'host_disk_reserve_bytes': 34359738368, 'application': [29, 1160], 'memcpy_channels': 16, 'fabric': [762, 1172], 'offset': [4, 1], 'stack_bytes': 4096, 'sram_ceiling_bytes': 48640}
def verify():
    raw=(ROOT/'manifest.json').read_bytes();admission=json.loads((ROOT/'admission.json').read_bytes())
    if admission!=dict(candidate=ROOT.name,manifest_sha256=hashlib.sha256(raw).hexdigest(),profile=PROFILE,compile_only=True,automatic_retry=False):
        raise ValueError('Exact complete Layer3 compile-only admission required')
    source=json.loads(raw)['files']
    for name,pin in source.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Source path')
        p=ROOT/name
        if p.is_symlink() or p.stat().st_size!=pin['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=pin['sha256']:
            raise ValueError('Frozen source changed: '+name)
    if {p.name for p in ROOT.glob('*.csl')}!={n for n in source if n.endswith('.csl')}:raise ValueError('Frozen CSL closure')
    if shutil.disk_usage(ROOT).free<PROFILE['host_disk_reserve_bytes']:raise ValueError('ALCF disk reserve')
    return admission
