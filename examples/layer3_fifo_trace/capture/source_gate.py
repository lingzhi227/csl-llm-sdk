"""Exact admission of one finite diagnostic original-layer physical capture."""
import hashlib
import json
from pathlib import Path
import shutil
from contract import summary
ROOT = Path(__file__).resolve().parent
COUNTS = summary(json.loads((ROOT/'layer3-plan.json').read_bytes()))
PROFILE = dict(runtime_seconds=600, capture_progress_seconds=30, worker_cpu_millicores=2000, worker_memory_bytes=4 << 30,
    host_address_space_bytes=4 << 30, host_sampled_rss_bytes=1 << 30, host_file_bytes=128 << 20,
    host_log_bytes=8 << 20, candidate_bytes=64 << 20, host_disk_reserve_bytes=32 << 30,
    input_count=1, copies=[COUNTS['copies']], launches=3, maximum_host_transfer_bytes=16 << 20,
    observer_delay_seconds=20, observer_deadline_seconds=35, observer_reads=1, unconditional_stop_after_snapshot=True, post_compute_all_PE_read=False, observed_heads=list(range(24)), observer_rectangle=[748, 0, 2, 25], observer_host_bytes=32000,
    application=[750, 45], fabric=[762, 1172], offset=[4, 1], diagnostic_only=True,
    mathematical_audit_during_allocation=False, between_phase_host_neural_feedback=False,
    full_model=False, automatic_retry=False)


def verify():
    raw = (ROOT/'manifest.json').read_bytes(); admission = json.loads((ROOT/'admission.json').read_bytes())
    expected = dict(candidate=ROOT.name, manifest_sha256=hashlib.sha256(raw).hexdigest(),
        profile=PROFILE, runtime_only=True, diagnostic_only=True, automatic_retry=False)
    if admission != expected:
        raise ValueError('Exact one-position physical all-head plus finite FIFO observer admission')
    for name, item in json.loads(raw)['files'].items():
        path = ROOT/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.is_file():
            raise ValueError('Frozen candidate path')
        if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Frozen source changed: '+name)
    if shutil.disk_usage(ROOT).free < PROFILE['host_disk_reserve_bytes']:
        raise ValueError('Native filesystem free reserve')
    return admission
