"""Read-only Linux preflight; no downloads, launches, cleanup, or reservations."""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import os
import shutil
import stat

GIB = 1024 ** 3


@dataclass(frozen=True)
class Budget:
    job_ram: int = 20 * GIB
    ram_reserve: int = 8 * GIB
    cache_limit: int = 20 * GIB
    disk_reserve: int = 32 * GIB
    max_files: int = 10000

    def __post_init__(self):
        for value in asdict(self).values():
            if type(value) is not int or value <= 0:
                raise ValueError('Budgets must be positive integer byte/count values')


def memory_available(text):
    for line in text.splitlines():
        fields = line.split()
        if fields and fields[0] == 'MemAvailable:':
            if len(fields) != 3 or fields[2] != 'kB':
                raise ValueError('Unsupported MemAvailable format')
            value = int(fields[1]) * 1024
            if value < 0:
                raise ValueError('Negative MemAvailable')
            return value
    raise ValueError('MemAvailable missing; refusing to guess from MemFree')


def cache_usage(root, max_entries):
    """Bounded, non-following traversal of an existing project-owned cache."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Cache must be an existing non-symlink directory')
    size = entries = 0
    pending = [root]
    while pending:
        with os.scandir(pending.pop()) as directory:
            for entry in directory:
                entries += 1
                if entries > max_entries:
                    raise ValueError('Cache entry limit exceeded')
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError('Cache symlinks are not allowed')
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    size += max(info.st_size, info.st_blocks * 512)
                else:
                    raise ValueError('Cache contains a special file')
    return size, entries


def assess(budget, available_ram, free_disk, cache_bytes, entries, planned_write):
    values = (available_ram, free_disk, cache_bytes, entries, planned_write)
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError('Measurements must be nonnegative integers')
    reasons = []
    if available_ram < budget.job_ram + budget.ram_reserve:
        reasons.append('Insufficient available RAM for job plus reserve')
    if cache_bytes + planned_write > budget.cache_limit:
        reasons.append('Projected cache exceeds cache budget')
    if free_disk < planned_write + budget.disk_reserve:
        reasons.append('Projected disk free space falls below reserve')
    if entries > budget.max_files:
        reasons.append('Cache entry limit exceeded')
    return {'admitted': not reasons, 'reasons': reasons,
            'scope': 'preflight_only_no_reservation_or_runtime_enforcement',
            'budget': asdict(budget),
            'observed': dict(zip(('available_ram', 'free_disk', 'cache_bytes',
                                 'entries', 'planned_write'), values))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--planned-write-bytes', type=int, required=True)
    args = parser.parse_args()
    try:
        budget = Budget()
        size, entries = cache_usage(args.cache, budget.max_files)
        available = memory_available(Path('/proc/meminfo').read_text())
        report = assess(budget, available, shutil.disk_usage(args.cache).free,
                        size, entries, args.planned_write_bytes)
    except (OSError, ValueError) as exc:
        print(json.dumps({'admitted': False, 'error': str(exc)}))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report['admitted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
