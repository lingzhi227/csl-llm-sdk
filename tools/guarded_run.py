"""Linux/systemd bounded serial execution. Owns only uniquely named project units."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.locking import heavy_lock
from qwen38.resources import Budget, GIB, assess, cache_usage, memory_available


def systemctl(*args):
    return subprocess.run(['systemctl', '--user', *args], text=True,
                          capture_output=True, check=True, timeout=15).stdout


def measure(cache, budget):
    size, entries = cache_usage(cache, budget.max_files)
    return memory_available(Path('/proc/meminfo').read_text()), shutil.disk_usage(cache).free, size, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--profile', choices=('sdk', 'cpu-reference'), default='sdk')
    args = parser.parse_args()
    cpu_reference = args.profile == 'cpu-reference'
    max_seconds = 60 if cpu_reference else 300
    memory_limit = 2 * GIB if cpu_reference else 20 * GIB
    cache, work = args.cache.resolve(), args.work.resolve()
    if not work.is_relative_to(cache) or cache == work:
        raise ValueError('Work directory must be inside the project cache')
    spec = json.loads(args.spec.read_text())
    if spec.get('required_profile', args.profile) != args.profile:
        raise ValueError('Execution profile differs from the frozen specification')
    if len(spec['steps']) not in (1, 2):
        raise ValueError('Only one or two serial steps are allowed')
    for step in spec['steps']:
        if type(step['seconds']) is not int or not 1 <= step['seconds'] <= max_seconds:
            raise ValueError(f'Step deadline must be 1..{max_seconds} seconds')
        if not step['argv'] or not all(type(x) is str for x in step['argv']):
            raise ValueError('argv must be a nonempty string array')
    budget = Budget(job_ram=memory_limit)
    receipt = work/'supervisor.json'
    if receipt.exists():
        raise ValueError('Run already attempted; use a newly reviewed frozen run')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'steps': [],
              'status': 'running', 'scope': 'bounded_microexperiment', 'profile': args.profile}
    def save():
        temporary = receipt.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(receipt)
    with heavy_lock(cache.parent/'qwen38-heavy.lock') as owner:
        # A supervisor crash releases flock; surviving units must still block relaunch.
        units = systemctl('list-units', 'qwen38-job-*.service', '--all', '--no-legend', '--plain')
        if units.strip():
            raise RuntimeError('Existing project unit requires inspection: '+units)
        report['owner'] = owner
        preflight = assess(budget, *measure(cache, budget), spec['planned_write_bytes'])
        report['preflight'] = preflight
        save()
        if not preflight['admitted']:
            report['status'] = 'preflight_refused'; save(); return 2
        for step in spec['steps']:
            current = assess(budget, *measure(cache, budget), spec['planned_write_bytes'])
            if not current['admitted']:
                report.update(status='between_steps_refused', refusal=current); save(); return 2
            unit = 'qwen38-job-'+uuid.uuid4().hex+'.service'
            item = {'name': step['name'], 'argv': step['argv'], 'unit': unit,
                    'deadline_seconds': step['seconds'], 'peak_cgroup_bytes': 0}
            report['steps'].append(item); save()
            start = time.monotonic()
            failure = None
            props = {}
            try:
                command = ['systemd-run', '--user', '--quiet', '--unit='+unit,
                           '--service-type=exec', '--working-directory='+str(work),
                           '-p', 'RemainAfterExit=yes', '-p', 'MemoryMax='+str(memory_limit),
                           '-p', 'MemorySwapMax=0', '-p', 'TasksMax=128',
                           '-p', 'CPUQuota='+('100%' if cpu_reference else '400%'), '-p', 'KillMode=control-group',
                           '-p', 'TimeoutStopSec=5', '-p', 'SendSIGKILL=yes',
                           '-p', 'LimitCORE=0', '-p', 'LimitFSIZE=67108864',
                           '-p', 'RuntimeMaxSec='+str(step['seconds']),
                           '-p', 'StandardOutput=append:'+str(work/(step['name']+'.log')),
                           '-p', 'StandardError=inherit',
                           '--setenv=TMPDIR='+str(work/'tmp'), '--setenv=OMP_NUM_THREADS=1',
                           '--setenv=OPENBLAS_NUM_THREADS=1',
                           '--setenv=MKL_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
                           *(['--setenv=CUDA_VISIBLE_DEVICES='] if cpu_reference else []), *step['argv']]
                (work/'tmp').mkdir(exist_ok=True)
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)
                while True:
                    output = systemctl('show', unit, '--property=ActiveState,SubState,Result,ExecMainStatus,ControlGroup,MemoryCurrent,MemoryPeak,MemoryMax,MemorySwapMax')
                    props = dict(line.split('=',1) for line in output.splitlines() if '=' in line)
                    item['properties'] = props
                    cg = props.get('ControlGroup')
                    if cg:
                        peak = Path('/sys/fs/cgroup'+cg)/'memory.peak'
                        if peak.exists():
                            item['peak_cgroup_bytes'] = max(item['peak_cgroup_bytes'], int(peak.read_text()))
                    for key in ('MemoryCurrent', 'MemoryPeak'):
                        value = props.get(key, '')
                        if value.isdecimal() and int(value) < 2**63:
                            item['peak_cgroup_bytes'] = max(item['peak_cgroup_bytes'], int(value))
                    available, free, size, entries = measure(cache, budget)
                    item['last_resources'] = dict(available_ram=available, free_disk=free, cache_bytes=size, entries=entries)
                    if available < budget.ram_reserve or free < budget.disk_reserve or size > budget.cache_limit:
                        raise RuntimeError('Runtime resource reserve lost')
                    if time.monotonic()-start > step['seconds'] + 5:
                        raise TimeoutError('Outer wall deadline exceeded')
                    if props.get('SubState') == 'exited' or props.get('ActiveState') in ('failed','inactive'):
                        break
                    save()
                    time.sleep(1)
                if props.get('Result') != 'success' or props.get('ExecMainStatus') != '0' or props.get('SubState') != 'exited':
                    raise RuntimeError('Service execution failed')
                item['execution'] = 'normal_exit'
            except BaseException as exc:
                failure = str(exc)
                item.update(execution='failed', error=failure)
            finally:
                item['wall_seconds'] = time.monotonic()-start
                try:
                    systemctl('stop', unit)
                    item['cleanup'] = 'unit_stopped'
                except Exception as exc:
                    item['cleanup'] = 'unconfirmed: '+str(exc)
                    failure = failure or item['cleanup']
                save()
            if failure:
                report['status'] = 'failed'; save(); return 1
        report['status'] = 'normal_exit'; save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
