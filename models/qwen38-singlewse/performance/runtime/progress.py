"""Bound post-load SDK stalls without shortening the separate image-load limit.

The supervisor observes progress from its own child. It retains the original
correlated-job cleanup path on timeout. This is diagnostic host instrumentation,
not a device completion signal or a serving-path timing implementation.
"""
import json
import os
from pathlib import Path
import time


class Progress:
    def __init__(self, enabled, path=Path('progress.json')):
        self.enabled=enabled;self.path=Path(path);self.sequence=0

    def mark(self, phase, epoch=None):
        if not self.enabled:return
        self.sequence+=1
        record=dict(pid=os.getpid(),sequence=self.sequence,phase=phase,epoch=epoch)
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record)+'\n');temporary.replace(self.path)
        print(json.dumps(dict(event='host_progress',**record)),flush=True)


class ProgressDeadline:
    def __init__(self, root, pid, seconds, clock=time.monotonic):
        if type(seconds) is not int or not 10<=seconds<=60:
            raise ValueError('Post-load progress deadline must be10..60 seconds')
        self.path=Path(root)/'progress.json';self.pid=pid;self.seconds=seconds
        self.clock=clock;self.sequence=0;self.since=None;self.phase=None

    def check(self):
        if not self.path.exists():
            if self.sequence:raise ValueError('Progress record disappeared')
            return  # Image loading retains the enclosing stage's hard limit.
        with self.path.open() as stream:
            raw=stream.read(4097)
        if len(raw)>4096:raise ValueError('Progress record exceeds bound')
        record=json.loads(raw);sequence=record['sequence']
        if record['pid']!=self.pid or type(sequence) is not int or sequence<1 or sequence<self.sequence:
            raise ValueError('Foreign or regressed progress record')
        now=self.clock()
        if sequence!=self.sequence:
            self.sequence=sequence;self.since=now;self.phase=record['phase']
        if now-self.since>=self.seconds:
            raise TimeoutError('Post-load progress deadline: '+self.phase)
