"""One scoped CPU0 simulation of immutable compiled outputs; no compile or retry."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from qwen38.locking import heavy_lock
from qwen38.resources import Budget, assess, cache_usage, memory_available
from qwen38.wp16_affinity import observe
from qwen38.resident_sdk_admission import (
    PROFILE, STEPS, THREAD_ENV, WORK, image_stamp, require, verify, verify_compiled, stage_limits)


def ctl(*args):
    return subprocess.run(['systemctl', '--user', *args], capture_output=True,
                          check=True, text=True, timeout=5).stdout


def properties(unit):
    names = 'LoadState,ActiveState,SubState,MainPID,ControlGroup,Result,ExecMainStatus,MemoryCurrent,MemoryPeak'
    return dict(line.split('=', 1) for line in ctl('show', unit, '--property=' + names).splitlines() if '=' in line)


def measure():
    size, entries = cache_usage(WORK.parent, 10000)
    return memory_available(Path('/proc/meminfo').read_text()), shutil.disk_usage(WORK.parent).free, size, entries


def check_output_bounds():
    """Sampled per-log/storage stops in addition to the2MiB hard per-file ceiling."""
    for path in WORK.rglob('*.log'):
        require(not path.is_symlink() and path.is_file() and path.stat().st_size <= PROFILE['log_bytes'], 'Individual SDK log budget exceeded')
    if (WORK/'out').exists():
        require(cache_usage(WORK/'out',512)[0] <= PROFILE['compiled_bytes'], 'Compiled directory budget exceeded')
    journal = WORK/'device-evidence/journal.jsonl'
    require(not journal.exists() or journal.stat().st_size <= PROFILE['device_journal_bytes'], 'Device journal byte budget exceeded')
    require(sum(path.stat().st_size for path in (WORK/'device-evidence').glob('*.npz')) <= PROFILE['private_observation_archive_bytes'], 'Private archive budget exceeded')


def main():
    require(Path.cwd() == WORK, 'Exact SDK working directory')
    require(not any((WORK / name).exists() for name in
                    ('supervisor.json', 'device-evidence', 'compile.log', 'simulate.log',
                     'runtime-admission-compile.json', 'runtime-admission-simulate.json', 'affinity.jsonl')),
            'SDK candidate already attempted')
    digest = verify(WORK)
    verify_compiled(WORK)
    image = image_stamp()
    require(0 in os.sched_getaffinity(0), 'CPU0 must be available before unit launch')
    report = dict(status='preparing', source_manifest_sha256=digest, image=image, steps=[],
                  launching_affinity=sorted(os.sched_getaffinity(0)), poll_seconds=.5)
    journal_bytes = 0
    journal_records = 0
    known = set()

    def save():
        temporary = WORK / 'supervisor.tmp'
        raw = json.dumps(report, indent=2) + '\n'
        require(len(raw.encode()) <= 262144, 'Bounded SDK supervisor receipt')
        with temporary.open('w') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(WORK / 'supervisor.json')

    def affinity_event(stage, unit, result):
        nonlocal journal_bytes, journal_records
        fresh = []
        for sample in result['samples']:
            identity = (sample['pid'], sample['tid'], sample['start_ticks'])
            if identity not in known:
                known.add(identity); fresh.append(sample)
        require(len(known) <= 8192, 'Bounded unique native task identity inventory')
        raw = (json.dumps(dict(record=journal_records, stage=stage, unit=unit,
            monotonic=time.monotonic(), checked_live_tasks=len(result['samples']),
            exited_during_observation=result['exited_during_observation'],
            all_observed_masks=[0], new_task_identities=fresh,
            samples_sha256=hashlib.sha256(json.dumps(result['samples'], sort_keys=True).encode()).hexdigest()), sort_keys=True) + '\n').encode()
        journal_records += 1; journal_bytes += len(raw)
        require(len(raw) <= 65536 and journal_records <= 1600 and journal_bytes <= PROFILE['affinity_journal_bytes'],
                'Bounded native affinity journal')
        with (WORK / 'affinity.jsonl').open('ab') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())

    with heavy_lock(WORK.parent.parent / 'qwen38-heavy.lock') as owner:
        require(not ctl('list-units', 'qwen38-job-*.service', '--all', '--no-legend', '--plain').strip(), 'Another heavy unit exists')
        report.update(owner=owner, status='running')
        save()
        for step in STEPS['steps']:
            try:
                verify(WORK)
                image_stamp()
                budget = Budget(job_ram=int(stage_limits(step['name'])['memory.max']))
                preflight = assess(budget, *measure(), PROFILE['ssd_run_bytes'])
                if step['name'] == 'simulate':
                    # Fresh candidate contains the exact preserved SDK001 compile.
                    require(not report['steps'], 'Only one simulation continuation')
                    verify_compiled(WORK)
                require(not ctl('list-units', 'qwen38-job-*.service', '--all', '--no-legend', '--plain').strip(), 'Heavy queue must be empty before each stage')
            except BaseException as exc:
                report.update(status='failed_before_next_unit', error_type=type(exc).__name__, message=str(exc))
                save(); return 1
            if not preflight['admitted']:
                report.update(status='preflight_failed_before_next_unit', preflight=preflight); save(); return 1
            unit = 'qwen38-job-' + uuid.uuid4().hex + '.service'
            fields = dict(RemainAfterExit='yes', MemoryMax=stage_limits(step['name'])['memory.max'], MemorySwapMax='0',
                TasksMax='128', RuntimeMaxSec=str(step['seconds']), LimitFSIZE=str(PROFILE['file_bytes']),
                LimitCORE='0', KillMode='control-group', TimeoutStopSec='2', SendSIGKILL='yes', CPUAffinity='0',
                StandardOutput='append:' + str(WORK / (step['name'] + '.log')), StandardError='inherit')
            command = ['systemd-run', '--user', '--quiet', '--unit=' + unit, '--service-type=exec',
                       '--working-directory=' + str(WORK)]
            for name, value in fields.items():
                command += ['-p', name + '=' + value]
            environment = dict(THREAD_ENV, WP16_SDK_UNIT=unit, WP16_SDK_STAGE=step['name'], TMPDIR=str(WORK / 'tmp'))
            command += ['--setenv=' + name + '=' + value for name, value in environment.items()]
            command += step['argv']
            item = dict(name=step['name'], unit=unit, status='running', preflight=preflight, fields=fields,
                argv=command, hard_seconds=step['seconds'], peak_cgroup_bytes=0,
                memory_peak_scope='maximum_observed_or_reported_sample_may_miss_final_peak',
                affinity_poll_count=0, affinity_live_sample_count=0, peak_work_directory_bytes=0,
                periodic_source_identity_checks=0)
            report['steps'].append(item); save()
            (WORK / 'tmp').mkdir(exist_ok=True)
            started = time.monotonic()
            last_source_check = started
            last_group = ''
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
                while True:
                    props = properties(unit)
                    item['properties'] = props
                    group_name = props.get('ControlGroup', '')
                    if group_name:
                        last_group = group_name
                        result = observe(group_name, [0])
                        item['affinity_poll_count'] += 1
                        item['affinity_live_sample_count'] += len(result['samples'])
                        affinity_event(step['name'], unit, result)
                        group = Path('/sys/fs/cgroup' + group_name)
                        try:
                            actual = {name: (group / name).read_text().strip() for name in ('memory.max', 'memory.swap.max', 'pids.max')}
                            require(actual == stage_limits(step['name']), 'Runtime actual SDK limits changed')
                            item['last_actual_limits'] = actual
                            item['peak_cgroup_bytes'] = max(item['peak_cgroup_bytes'], int((group / 'memory.peak').read_text()))
                            item['last_memory_events'] = (group / 'memory.events').read_text()
                            item['last_pids_events'] = (group / 'pids.events').read_text()
                            for text, keys in [(item['last_memory_events'], ('max','oom','oom_kill','oom_group_kill')),
                                               (item['last_pids_events'], ('max',))]:
                                events = dict(line.split() for line in text.splitlines())
                                require(all(key in events and int(events[key]) == 0 for key in keys), 'SDK memory or task pressure')
                        except FileNotFoundError:
                            pass  # Unit may finish between its property and cgroup reads.
                    for name in ('MemoryCurrent', 'MemoryPeak'):
                        value = props.get(name, '')
                        if value.isdecimal() and int(value) < 2**63:
                            item['peak_cgroup_bytes'] = max(item['peak_cgroup_bytes'], int(value))
                    require(time.monotonic() - started < step['seconds'], 'Scoped SDK wall deadline exceeded')
                    available, free, cache_bytes, entries = measure()
                    item['last_resources'] = dict(available_ram=available, free_disk=free, cache_bytes=cache_bytes, entries=entries)
                    require(available >= PROFILE['ram_reserve_bytes'] and free >= PROFILE['ssd_reserve_bytes'] and
                            cache_bytes <= PROFILE['ssd_cache_bytes'], 'Runtime global resource reserve crossed')
                    run_bytes, _ = cache_usage(WORK, 1024)
                    item['peak_work_directory_bytes'] = max(item['peak_work_directory_bytes'], run_bytes)
                    require(run_bytes <= PROFILE['ssd_run_bytes'], 'SDK candidate directory budget exceeded')
                    check_output_bounds()
                    if time.monotonic() - last_source_check >= 5:
                        verify(WORK)
                        item['periodic_source_identity_checks'] += 1
                        last_source_check = time.monotonic(); save()
                    if props.get('SubState') == 'exited' or props.get('ActiveState') in ('failed', 'inactive'):
                        break
                    time.sleep(.5)
                require(props.get('Result') == 'success' and props.get('ExecMainStatus') == '0' and
                        props.get('SubState') == 'exited', 'SDK stage unit failed')
                exit_record = json.loads((WORK / ('container-exit-' + step['name'] + '.json')).read_bytes())
                require(exit_record['status'] == 'passed' and exit_record['affinity'] == [0], 'Normal container exit and CPU0')
                require(item['affinity_live_sample_count'] > 0 and item.get('last_actual_limits') ==
                        stage_limits(step['name']), 'Positive native mask and actual cgroup observations')
                if step['name'] == 'compile':
                    verify_compiled(WORK)
                else:
                    outcome = json.loads((WORK / 'result.json').read_bytes())
                    require(outcome['status'] == 'passed' and outcome['normal_stop'] is True, 'All SDK gates and normal stop required')
                    item['post_runtime_compiled_inventory'] = verify_compiled(WORK, after_runtime=True)
                verify(WORK)
                image_stamp()
                item['status'] = 'passed'
            except BaseException as exc:
                item.update(status='failed', error_type=type(exc).__name__, message=str(exc))
            finally:
                item['seconds'] = time.monotonic() - started
                try:
                    ctl('stop', unit)
                    after = properties(unit)
                    require(after['MainPID'] == '0' and after['ActiveState'] in ('inactive', 'failed'), 'Owned SDK unit not quiescent')
                    if last_group:
                        procs = Path('/sys/fs/cgroup' + last_group) / 'cgroup.procs'
                        require(not procs.exists() or not procs.read_text().strip(), 'Owned SDK cgroup still populated')
                    if after['ActiveState'] == 'failed':
                        ctl('reset-failed', unit)
                    item.update(cleanup='owned_unit_quiescent', cleanup_properties=properties(unit), last_owned_cgroup=last_group)
                except BaseException as exc:
                    item.update(status='failed', cleanup='unconfirmed_' + type(exc).__name__, cleanup_message=str(exc))
                else:
                    try:
                        check_output_bounds()
                        if step['name'] == 'simulate':
                            item['post_cleanup_compiled_inventory'] = verify_compiled(WORK, after_runtime=True)
                    except BaseException as exc:
                        item.update(status='failed', post_cleanup_inventory_error=type(exc).__name__, post_cleanup_message=str(exc))
                report.update(affinity_journal_bytes=journal_bytes, affinity_journal_records=journal_records,
                              distinct_native_task_identities=len(known))
                save()
            if item['status'] != 'passed':
                report['status'] = 'failed'; save(); return 1
        report['status'] = 'passed'; save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
