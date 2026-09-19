"""Exact separately admitted physical runtime with frozen source identity."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILE={'runtime_seconds': 300, 'worker_cpu_millicores': 1000, 'worker_memory_bytes': 4294967296, 'host_address_space_bytes': 4294967296, 'host_sampled_rss_bytes': 1073741824, 'host_file_bytes': 134217728, 'host_log_bytes': 8388608, 'candidate_bytes': 33554432, 'host_disk_reserve_bytes': 34359738368, 'application': [4, 1], 'fabric': [762, 1172], 'offset': [4, 1], 'copies': 13, 'H2D': 0, 'D2H': 13, 'launches': 6, 'host_slot_bytes': 32768, 'reset_generations': 2, 'negative_calls': 61, 'positive_calls': 172, 'raw_captures': 13, 'journal_events': 43, 'full_model': False}
def verify():
    raw=(ROOT/'manifest.json').read_bytes()
    expected=dict(candidate=ROOT.name,manifest_sha256=hashlib.sha256(raw).hexdigest(),profile=PROFILE,runtime_only=True,simulator=False,automatic_retry=False)
    if json.loads((ROOT/'admission.json').read_bytes())!=expected:
        raise ValueError('Exact four-PE physical runtime admission')
    for name,spec in json.loads(raw)['files'].items():
        path=ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.is_file():
            raise ValueError('Safe frozen source path')
        raw=path.read_bytes()
        if len(raw)!=spec['bytes'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:
            raise ValueError('Frozen source identity: '+name)
    if shutil.disk_usage(ROOT).free<PROFILE['host_disk_reserve_bytes']:
        raise ValueError('ALCF disk reserve')
    return expected
