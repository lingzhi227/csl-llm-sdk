"""Bounded advisory diagnostics using anonymous memory and a nonblocking pipe.

No model data, locals, filesystem writes, heartbeat deadline renewal, or SDK
calls. The independently owned parent retains diagnostics through cleanup.
An unavailable/torn sample never proves progress or completion.
"""
from contextlib import contextmanager
import faulthandler
import json
import mmap
import os
import signal
import struct
import time

PAGE = 4096
MAX_STACK_BYTES = 256 << 10
MAX_STACK_REQUESTS = 4
STACK_AFTER_SECONDS = 8.
_child = None


class Child:
    def __init__(self, memory_fd, stack_fd):
        self.memory = mmap.mmap(memory_fd, PAGE)
        self.stack_fd = stack_fd
        self.sequence = 0
        self.state = dict(scope='startup', operation='entry', label='', durable_events=0,
                          durable_bytes=0, written_events=0, written_bytes=0)
        faulthandler.register(signal.SIGUSR2, file=stack_fd, all_threads=True, chain=False)
        self.mark('entry')

    def mark(self, operation, label='', **fields):
        self.state.update(fields, operation=str(operation)[:96], label=str(label)[:256],
                          monotonic=time.monotonic())
        raw = json.dumps(self.state, separators=(',', ':')).encode()
        if len(raw) > PAGE - 16:
            raise ValueError('Bounded advisory operation state')
        self.sequence += 2
        # Odd/even version brackets reject interrupted concurrent publication.
        struct.pack_into('<Q', self.memory, 0, self.sequence - 1)
        struct.pack_into('<Q', self.memory, 8, len(raw))
        self.memory[16:16 + len(raw)] = raw
        struct.pack_into('<Q', self.memory, 0, self.sequence)

    def close(self):
        faulthandler.unregister(signal.SIGUSR2)
        self.memory.close()


def attach():
    global _child
    if _child is not None:
        raise ValueError('Exactly one owned advisory channel')
    _child = Child(int(os.environ['CHAIN_LIVENESS_FD']), int(os.environ['CHAIN_STACK_FD']))
    return _child


def mark(operation, label='', **fields):
    if _child is not None:
        _child.mark(operation, label, **fields)


@contextmanager
def operation(name, label=''):
    previous = None if _child is None else dict(_child.state)
    mark(name, label)
    try:
        yield
    finally:
        if previous is not None:
            mark(previous['operation'], previous['label'])


class Parent:
    def __init__(self):
        self.fd = os.memfd_create('chain-operation', os.MFD_CLOEXEC)
        os.ftruncate(self.fd, PAGE)
        self.memory = mmap.mmap(self.fd, PAGE)
        self.read_fd, self.write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        self.stack = bytearray()
        self.requests = []
        self.last = None
        self.truncated = False
        self.total_pipe_bytes = 0

    def environment(self):
        return dict(CHAIN_LIVENESS_FD=str(self.fd), CHAIN_STACK_FD=str(self.write_fd))

    def child_started(self):
        os.close(self.write_fd)
        self.write_fd = None

    def snapshot(self):
        first = struct.unpack_from('<Q', self.memory, 0)[0]
        if not first or first % 2:
            return None
        size = struct.unpack_from('<Q', self.memory, 8)[0]
        if size > PAGE - 16:
            raise ValueError('Bounded independent state size')
        raw = self.memory[16:16 + size]
        if first != struct.unpack_from('<Q', self.memory, 0)[0]:
            return None
        value = json.loads(raw)
        value['publication'] = first
        self.last = value
        return value

    def drain(self):
        # Bound each poll's work even if an unexpected writer fills the pipe.
        for _ in range(8):
            try:
                raw = os.read(self.read_fd, 16384)
            except BlockingIOError:
                break
            if not raw:
                break
            self.total_pipe_bytes += len(raw)
            available = MAX_STACK_BYTES - len(self.stack)
            self.stack.extend(raw[:available])
            self.truncated |= len(raw) > available

    def poll(self, pid, now=None):
        self.drain()
        value = self.snapshot()
        now = time.monotonic() if now is None else now
        if (value is not None and value['scope'] == 'capture' and
                now - value['monotonic'] >= STACK_AFTER_SECONDS and
                len(self.requests) < MAX_STACK_REQUESTS and
                all(r['publication'] != value['publication'] for r in self.requests)):
            try:
                os.kill(pid, signal.SIGUSR2)
            except ProcessLookupError:
                return
            self.requests.append(dict(publication=value['publication'], monotonic=now,
                                      operation=value['operation'], label=value['label']))

    def report(self):
        self.drain()
        self.snapshot()
        return dict(schema=1, transport='anonymous shared memory and nonblocking signal pipe',
                    last_operation=self.last, requests=self.requests,
                    stack_text=self.stack.decode(errors='replace'), retained_stack_bytes=len(self.stack),
                    pipe_bytes_read=self.total_pipe_bytes, retained_output_truncated=self.truncated,
                    signal_pipe_output_may_be_partial=True, deadline_renewal=False,
                    filesystem_dependency=False, success_or_durability_proof=False)

    def close(self):
        self.memory.close()
        os.close(self.fd)
        os.close(self.read_fd)
        if self.write_fd is not None:
            os.close(self.write_fd)
