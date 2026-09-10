"""Read actual singleton CPU masks; never change a task's affinity."""
import os
from pathlib import Path


def current(expected=(0,)):
    mask = sorted(os.sched_getaffinity(0))
    if mask != list(expected):
        raise ValueError('current_task_affinity_mismatch')
    return mask


def start_ticks(path):
    return path.read_text().rsplit(')', 1)[1].split()[19]


def observe(group_path, expected=(0,), *, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup')):
    if not group_path.startswith('/') or '..' in Path(group_path).parts:
        raise ValueError('invalid_owned_cgroup_path')
    group = cgroup_root / group_path.lstrip('/')
    result = dict(method='single_logical_cpu_affinity', expected=list(expected),
                  cpu_max_present=(group / 'cpu.max').exists(), samples=[], exited_during_observation=0)
    try:
        pids = (group / 'cgroup.procs').read_text().split()
    except FileNotFoundError:
        return result
    if len(pids) > 128 or any(not p.isdecimal() for p in pids):
        raise ValueError('unexpected_cgroup_process_inventory')
    for pid_text in pids:
        pid = int(pid_text)
        process = proc_root / pid_text
        try:
            if '0::' + group_path not in (process / 'cgroup').read_text().splitlines():
                result['exited_during_observation'] += 1
                continue
            tasks = list((process / 'task').iterdir())
        except FileNotFoundError:
            result['exited_during_observation'] += 1
            continue
        if len(tasks) > 128:
            raise ValueError('unexpected_thread_count')
        for task in tasks:
            tid = int(task.name)
            try:
                before = start_ticks(task / 'stat')
                mask = sorted(os.sched_getaffinity(tid))
                after = start_ticks(task / 'stat')
                membership = (process / 'cgroup').read_text().splitlines()
            except (FileNotFoundError, ProcessLookupError):
                result['exited_during_observation'] += 1
                continue
            if before != after or '0::' + group_path not in membership:
                result['exited_during_observation'] += 1
                continue
            if mask != list(expected):
                raise ValueError('live_job_task_affinity_mismatch')
            result['samples'].append(dict(pid=pid, tid=tid, start_ticks=before, mask=mask))
    return result
