"""Host-side actual cgroup/CPU0 admission before entering the existing SDK image."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from qwen38.wp16_affinity import current
from qwen38.resident_sdk_admission import (
    IMAGE, PROFILE, THREAD_ENV, WORK, image_stamp, require, verify, verify_compiled, stage_limits)


def main():
    require(len(sys.argv) == 2 and sys.argv[1] == 'simulate', 'Exact simulation-only SDK stage')
    stage = sys.argv[1]
    require(Path.cwd() == WORK and os.environ.get('WP16_SDK_STAGE') == stage, 'Owned SDK stage and directory')
    digest = verify(WORK)
    affinity = current()
    unit = os.environ.get('WP16_SDK_UNIT', '')
    groups = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(unit.startswith('qwen38-job-') and len(groups) == 1 and groups[0].endswith('/' + unit), 'Owned SDK unit cgroup')
    group = Path('/sys/fs/cgroup' + groups[0])
    actual = {name: (group / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'pids.max')}
    require(actual == stage_limits(stage), 'Actual SDK hard limits')
    require(all(os.environ.get(k) == v for k, v in THREAD_ENV.items()), 'Explicit single-worker host environment')
    image = image_stamp()
    if stage == 'simulate':
        verify_compiled(WORK)
    temporary = WORK / 'tmp'
    require(temporary.is_dir() and not temporary.is_symlink(), 'Task-owned SDK temporary directory')
    environment = dict(THREAD_ENV, TMPDIR='/tmp', WP16_SDK_STAGE=stage, WP16_SDK_UNIT=unit)
    command = ['/usr/local/bin/singularity', 'exec', '-C', '--bind=' + str(WORK) + ':' + str(WORK),
               '--bind=' + str(temporary) + ':/tmp', '--pwd=' + str(WORK)]
    command += ['--env=' + name + '=' + value for name, value in environment.items()]
    command += [str(IMAGE), '/python/python-x86_64/bin/python', '-B', 'container_entry.py', stage]
    receipt = dict(stage=stage, unit=unit, source_manifest_sha256=digest, actual_limits=actual,
        cgroup=groups[0], affinity=affinity, cpu_max_present=(group / 'cpu.max').exists(),
        sdk_import_has_not_started=True, image=image, command=command)
    with (WORK / ('runtime-admission-' + stage + '.json')).open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.execv(command[0], command)


if __name__ == '__main__':
    main()
