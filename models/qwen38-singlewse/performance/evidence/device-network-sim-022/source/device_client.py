"""Bounded inline gateway frames; no host neural computation.

The SDK stream may contain many writes/admissions. Device IQ backpressure owns
each frame until its acknowledged operation retires. Read replies must be last
and are copied before another submission can overwrite the shared reply arena.
An operand acknowledgement means producer DMA completed, not projection drain.
"""
from dataclasses import dataclass
import numbers
import time
import numpy as np


def uint(value, maximum=0xffffffff):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= int(value) <= maximum:
        raise ValueError('Unsigned protocol value outside its declared width')
    return int(value)


@dataclass(frozen=True)
class Frame:
    wire: tuple
    audit: tuple
    reply: tuple

    @property
    def read_values(self):
        return self.reply[2] if self.reply else 0


class Encoder:
    def __init__(self, targets):
        self.targets = uint(targets, 65534)
        if not self.targets:
            raise ValueError('At least one internal target is required')
        self.total = 0
        self.sequences = {}

    def _frame(self, target, body, replies, audit, reply):
        if self.total == 0xffffffff:
            raise ValueError('Gateway sequence exhausted; reset the runtime')
        self.total += 1
        return Frame((target | 0x80000000, len(body), replies, self.total, *body),
                     (self.total, target, *audit), tuple(reply))

    def command(self, target, opcode, a=0, b=0, c=0, d=0, payload=()):
        target, opcode = uint(target, self.targets - 1), uint(opcode, 6)
        a, b, c, d = (uint(v) for v in (a, b, c, d))
        payload = tuple(uint(v) for v in payload)
        if opcode < 2:
            if not 1 <= c <= 256 or b + c >= 65536:
                raise ValueError('Transfer extent')
            if opcode == 0 and (len(payload) != c or (a == 1 and any(v > 65535 for v in payload))):
                raise ValueError('Write body extent or halfword width')
            if opcode == 1 and a == 1 and (b % 2 or c % 2):
                raise ValueError('Halfword reads require aligned pairs')
        if opcode != 0 and payload:
            raise ValueError('Only writes have a body')
        sequence = self.sequences.get(target, 0) + 1
        if sequence > 0xffffffff:
            raise ValueError('Endpoint sequence exhausted; reset the runtime')
        count = c if opcode == 1 else 0
        bits = 16 if count and a == 1 else 32
        reply = (sequence, 0, count, bits)
        body = (sequence, opcode, a, b, c, d, *payload)
        frame = self._frame(target, body, 4 + count // (2 if bits == 16 else 1), (sequence, opcode), reply)
        self.sequences[target] = sequence
        return frame

    def operand(self, invocation, group, parts, packed_bf16):
        invocation, group, parts = (uint(v) for v in (invocation, group, parts))
        words = tuple(uint(v) for v in packed_bf16)
        if not invocation or parts not in (40, 48) or group >= parts or len(words) != 64:
            raise ValueError('Original group128 operand shape or identity')
        return self._frame(self.targets, (invocation, group, parts, *words), 0, (invocation, 7), ())


def batches(frames, word_limit=16384):
    """Keep payload staging bounded; a read reply ends its batch."""
    word_limit = uint(word_limit, 262144)
    if word_limit < 266:
        raise ValueError('Batch cannot fit a maximum-size command')
    pending, words = [], 0
    for frame in frames:
        if pending and words + len(frame.wire) > word_limit:
            yield pending
            pending, words = [], 0
        pending.append(frame)
        words += len(frame.wire)
        if frame.read_values:
            yield pending
            pending, words = [], 0
    if pending:
        yield pending


class Client:
    def __init__(self, runner, dtype, order, gateway, completion_timeout=30, progress=None):
        if not 0 < completion_timeout <= 1200:
            raise ValueError('Finite completion timeout required')
        self.runner, self.gateway = runner, tuple(gateway)
        self.opts = dict(streaming=False, order=order.ROW_MAJOR, nonblock=False, data_type=dtype.MEMCPY_32BIT)
        self.ids = {n:runner.get_id(n) for n in ('device_audit', 'device_response')}
        self.completed = 0
        self.pending_reply = None
        self.timeout = completion_timeout
        self.stream_calls = 0
        self.progress = progress

    def _read(self, name, count):
        data = np.empty(count, np.uint32)
        self.runner.memcpy_d2h(data, self.ids[name], *self.gateway, 1, 1, count, **self.opts)
        return data

    def stream(self, frames):
        frames = list(frames)
        if not frames or self.pending_reply is not None:
            raise ValueError('Empty submission or unread prior reply')
        if sum(len(f.wire) for f in frames) > 262144 or any(f.read_values for f in frames[:-1]):
            raise ValueError('Unbounded batch or overwritten read reply')
        if [f.audit[0] for f in frames] != list(range(self.completed + 1, self.completed + 1 + len(frames))):
            raise ValueError('Gateway command gap, replay or reordering')
        data = np.asarray([v for f in frames for v in f.wire], np.uint32)
        self.runner.memcpy_h2d(14, data, *self.gateway, 1, 1, len(data), **dict(self.opts, streaming=True))
        self.stream_calls += 1
        last, deadline = frames[-1], time.monotonic() + self.timeout
        previous = self.completed
        while True:
            status = self._read('device_audit', 4)
            observed = int(status[0])
            if self.progress is not None and observed != previous:
                self.progress(dict(completed=observed,expected=last.audit[0],audit=status.tolist()))
            if not previous <= observed <= last.audit[0]:
                raise RuntimeError('Nonmonotone gateway completion')
            if observed == last.audit[0]:
                np.testing.assert_array_equal(status, last.audit)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Gateway command completion: observed '+str(status.tolist())+' expected '+str(last.audit))
            previous = observed
        self.completed = last.audit[0]
        if last.read_values:
            self.pending_reply = last

    def read_reply(self):
        frame = self.pending_reply
        if frame is None:
            raise ValueError('No unread data reply')
        count, bits = frame.reply[2:]
        wire = self._read('device_response', 4 + count // (2 if bits == 16 else 1))
        np.testing.assert_array_equal(wire[:4], frame.reply)
        self.pending_reply = None
        return wire[4:].view(np.uint16) if bits == 16 else wire[4:]
