"""Parent-side sampled deadlines, independent of a blocked SDK client.

The 0.1 second poll target is not a hard OS scheduling guarantee. A successful
capture separately proves coherent observer, every-PE completion, and two
complete transport fences. A blocked SDK call is handled by this parent, not
by an observer that cannot be read while that same call is blocked.
"""
import json
import math
from pathlib import Path
import time
from upload_monitor import Parent
from runtime_startup import Parent as StartupParent


def record(path):
    if not path.exists():return None
    if path.is_symlink() or path.stat().st_size>4096:raise ValueError('Small regular monitor record')
    value=json.loads(path.read_bytes())
    if not isinstance(value,dict):raise ValueError('Monitor record object')
    return value


def stamp(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Finite monotonic timestamp')
    return value


class Monitor:
    def __init__(self,root,profile,started,clock=None):
        self.root=Path(root);self.profile=profile;self.started=started
        self.clock=time.monotonic if clock is None else clock
        self.phase=None;self.phase_started=None;self.phase_index=-1
        self.max_journal_idle=0.;self.max_poll_gap=0.;self.last_poll=started
        self.journal_size=None;self.journal_advanced=None
        self.compute_windows={};self.latest_compute_serial=0
        self.weight_parent=Parent(root,started,profile['weight_upload_batches'],clock=self.clock)
        self.startup_parent=StartupParent(root,profile,started,clock=self.clock)

    def check(self,now):
        self.max_poll_gap=max(self.max_poll_gap,now-self.last_poll);self.last_poll=now
        if now-self.started>self.profile['runtime_seconds']:raise TimeoutError('Total physical client deadline')
        window=record(self.root/'monitor-deadline.json')
        # A child can publish between the caller's sample and this read.
        # Validate snapshot timestamps against a fresh sample AFTER each read.
        now=self.clock()
        compute_active=window is not None and window['active'] is True
        if window is not None and type(window['active']) is not bool:raise ValueError('Boolean compute active flag')
        compute_deadline=None
        if compute_active:
            start=stamp(window['started']);deadline=stamp(window['deadline'])
            if window['kind']!='compute_observer' or abs(deadline-start-self.profile['compute_observer_seconds'])>1e-6:
                raise ValueError('Exact compute/observer deadline')
            if start<self.started or start>now:raise ValueError('Compute start bounds')
            serial=window['serial']
            if type(serial) is not int or not 1<=serial<=3 or serial<self.latest_compute_serial:
                raise ValueError('Monotonic finite compute serial')
            identity=(start,deadline)
            if serial in self.compute_windows and self.compute_windows[serial]!=identity:
                raise ValueError('A compute deadline cannot be renewed')
            self.compute_windows[serial]=identity;self.latest_compute_serial=serial;compute_deadline=deadline
            if now>deadline:raise TimeoutError('Compute/observer deadline exceeded; terminate and release, not completion')
        phase=record(self.root/'runtime-phase.json')
        now=self.clock()
        if phase is None:
            self.startup_parent.check(None)
            if now-self.started>self.profile['startup_seconds']:raise TimeoutError('Pre-runtime startup deadline')
            return
        name=phase['phase'];order=('startup','capture','stop','stopped')
        if name not in order or order.index(name)<self.phase_index:raise ValueError('Monotonic runtime phase order')
        self.startup_parent.check(name)
        if name=='stopped':
            if phase.get('stopped') is not True:raise ValueError('Stopped phase flag')
            self.phase_index=3;self.phase=name;return
        start=stamp(phase['started']);deadline=stamp(phase['deadline'])
        seconds=self.profile[{'startup':'startup_seconds','capture':'capture_seconds','stop':'normal_stop_seconds'}[name]]
        if start<self.started or start>now or abs(deadline-start-seconds)>1e-6:raise ValueError('Exact runtime phase deadline')
        if name==self.phase and start!=self.phase_started:raise ValueError('A phase deadline cannot be renewed')
        self.phase=name;self.phase_started=start;self.phase_index=order.index(name)
        if now>deadline:raise TimeoutError(name+' absolute phase deadline')
        if name=='startup' and now-self.started>self.profile['startup_seconds']:
            raise TimeoutError('Startup including preparation deadline')
        if name=='capture':
            weight_active=self.weight_parent.check()
            if weight_active and compute_active:raise ValueError('Weight upload cannot overlap device compute')
            progress=record(self.root/'progress.json')
            now=self.clock()
            if now>deadline:raise TimeoutError('capture absolute phase deadline')
            latest=start if progress is None else stamp(progress['monotonic'])
            if latest<start or latest>now:raise ValueError('Capture progress timestamp')
            if progress is not None and (progress['raw_bytes']>self.profile['max_raw_bytes'] or progress['raw_files']>self.profile['max_raw_files']):raise ValueError('Durable raw byte/file cap')
            # Blocking compute/observer calls use their one nonrenewable
            # absolute window. All other capture calls retain the15s limit.
            if compute_active and now>compute_deadline:raise TimeoutError('Compute/observer deadline exceeded; terminate and release, not completion')
            if not compute_active and not weight_active and now-latest>self.profile['capture_progress_seconds']:raise TimeoutError('Blocked capture operation/progress deadline')
            journal=self.root/'journal.jsonl'
            if not journal.is_file():raise ValueError('Capture journal is required')
            size=journal.stat().st_size
            if size>self.profile['max_journal_bytes']:raise ValueError('Capture journal byte bound')
            if self.journal_size!=size:
                if self.journal_size is not None and size<self.journal_size:raise ValueError('Capture journal shrank')
                self.journal_size=size;self.journal_advanced=now
            else:
                idle=now-self.journal_advanced;self.max_journal_idle=max(self.max_journal_idle,idle)
                if not compute_active and not weight_active and idle>self.profile['capture_progress_seconds']:raise TimeoutError('Capture journal progress deadline')
