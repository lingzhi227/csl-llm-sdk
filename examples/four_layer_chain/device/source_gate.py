"""Exact once-only static chain compile; no runtime authorization."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILE={'compile_seconds': 1800, 'worker_cpu_millicores': 1000, 'worker_memory_bytes': 8589934592, 'host_address_space_bytes': 4294967296, 'host_sampled_rss_bytes': 1073741824, 'host_file_bytes': 134217728, 'host_log_bytes': 8388608, 'candidate_bytes': 536870912, 'host_disk_reserve_bytes': 34359738368, 'application': [119, 1160], 'memcpy_channels': 16, 'fabric': [762, 1172], 'offset': [4, 1], 'stack_bytes': 4096, 'sram_ceiling_bytes': 49152, 'legacy_family_ceilings': {'linear': 48128, 'attention': 48640}, 'maximum_ELFs': 8192, 'archive_members': 16384, 'archive_uncompressed_bytes': 805306368}
def verify():
 raw=(ROOT/'manifest.json').read_bytes();a=json.loads((ROOT/'admission.json').read_bytes())
 if a!=dict(candidate=ROOT.name,manifest_sha256=hashlib.sha256(raw).hexdigest(),profile=PROFILE,compile_only=True,automatic_retry=False):raise ValueError('Exact chain compile admission required')
 source=json.loads(raw)['files']
 for n,pin in source.items():
  p=ROOT/n
  if Path(n).is_absolute() or '..' in Path(n).parts or p.is_symlink() or dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())!=pin:raise ValueError('Frozen source identity: '+n)
 assert {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.csl')}=={n for n in source if n.endswith('.csl')}
 if shutil.disk_usage(ROOT).free<PROFILE['host_disk_reserve_bytes']:raise ValueError('ALCF disk reserve')
 return a
