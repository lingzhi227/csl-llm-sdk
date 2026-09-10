"""Cross-process lock with durable owner identity; never remove a lock inode."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket


def identity():
    text = Path('/proc/self/stat').read_text()
    return {'host': socket.gethostname(), 'pid': os.getpid(),
            'start_ticks': text.rsplit(')', 1)[1].split()[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


@contextmanager
def heavy_lock(path):
    path = Path(path)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            stream.seek(0)
            raise RuntimeError('Heavy task lock held: ' + stream.read(4096)) from None
        try:
            owner = identity()
            stream.seek(0)
            stream.truncate()
            json.dump(owner, stream)
            stream.flush()
            os.fsync(stream.fileno())
            yield owner
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
