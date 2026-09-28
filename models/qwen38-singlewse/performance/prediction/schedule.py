"""Deterministic closed-request pipeline with finite stage buffers.

Latency and initiation interval are separate. A finished item holds its slot
until the next stage accepts it. Feedback creates the next position of the same
request only after an actual modeled head completion. This queueing abstraction
does not emulate CSL instructions, packet-level credits, prefill or EOS/reset.
"""
from collections import deque
import heapq
import math


def simulate(stages, concurrency, *, feedback_cycles=0, warmup=100, samples=1000,
             clock_hz=None):
    if not stages or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError('Nonempty stages and positive integer concurrency required')
    if not isinstance(feedback_cycles,(int,float)) or feedback_cycles < 0 or not math.isfinite(feedback_cycles):
        raise ValueError('Invalid feedback latency')
    if warmup < 1 or samples < 2:
        raise ValueError('Positive warmup and at least two samples required')
    if clock_hz is not None and (not math.isfinite(clock_hz) or clock_hz <= 0):
        raise ValueError('Positive explicit clock required')
    ids = [s['id'] for s in stages]
    if len(ids) != len(set(ids)):
        raise ValueError('Stage IDs must be unique')
    for s in stages:
        for key in ('latency_cycles', 'ii_cycles'):
            if not isinstance(s[key],(int,float)) or not math.isfinite(s[key]) or s[key] <= 0:
                raise ValueError('Every modeled latency and II must be positive')
        if not isinstance(s['slots'], int) or s['slots'] < 1:
            raise ValueError('Positive stage slot count required')
    count = len(stages)
    queues = [deque() for _ in stages]
    flying = [dict() for _ in stages]
    finished = [deque() for _ in stages]
    next_start = [0.0]*count
    launches = [0]*count
    blocked = [0.0]*count
    peak = [0]*count
    events = []
    pending = deque()
    times = []
    per_request = [[] for _ in range(concurrency)]
    serial = 0

    def push(time, kind, index, job):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (time, serial, kind, index, job))

    def occupancy(i):
        return len(queues[i])+len(flying[i])

    for request in range(concurrency):
        push(0, 'input', 0, (request, 0))
    now = 0.0
    event_count = 0
    limit = max(10000, (warmup+samples+concurrency)*count*30)
    while len(times) < warmup+samples:
        if not events or event_count > limit:
            raise ValueError('Queue model stalled or exceeded its event budget')
        now = events[0][0]
        while events and events[0][0] == now:
            _, _, kind, i, job = heapq.heappop(events)
            event_count += 1
            if kind == 'input':
                pending.append(job)
            elif kind == 'done':
                flying[i][job] = now
                finished[i].append(job)
        changed = True
        while changed:
            changed = False
            # Moving downstream first releases backpressure toward the input.
            for i in reversed(range(count)):
                while finished[i] and (i == count-1 or occupancy(i+1) < stages[i+1]['slots']):
                    job = finished[i].popleft()
                    blocked[i] += now-flying[i].pop(job)
                    if i == count-1:
                        request, position = job
                        if len(per_request[request]) != position:
                            raise ValueError('Request dependency/order violation')
                        times.append(now)
                        per_request[request].append(now)
                        push(now+feedback_cycles, 'input', 0, (request, position+1))
                    else:
                        queues[i+1].append(job)
                    changed = True
            while pending and occupancy(0) < stages[0]['slots']:
                queues[0].append(pending.popleft())
                changed = True
            for i, s in enumerate(stages):
                peak[i] = max(peak[i], occupancy(i))
                if queues[i] and next_start[i] <= now:
                    job = queues[i].popleft()
                    flying[i][job] = None
                    launches[i] += 1
                    next_start[i] = now+s['ii_cycles']
                    push(now+s['latency_cycles'], 'done', i, job)
                    push(next_start[i], 'wake', i, None)
                    changed = True
    start = times[warmup-1]
    end = times[warmup+samples-1]
    if end <= start:
        raise ValueError('Measurement window has zero duration')
    rate = samples/(end-start)
    total_latency = sum(s['latency_cycles'] for s in stages)+feedback_cycles
    capacity = min(min(1/s['ii_cycles'], s['slots']/s['latency_cycles']) for s in stages)
    envelope = min(capacity, concurrency/total_latency)
    request_intervals = [b-a for ts in per_request for a,b in zip(ts,ts[1:]) if b > start]
    return dict(scope='Conditional finite-buffer queue simulation, not neural or fabric execution',
                concurrency=concurrency, first_completion_cycles=times[0],
                window_cycles=[start,end], completions_in_window=samples,
                generated_completions_per_cycle=rate,
                generated_tokens_per_second=rate*clock_hz if clock_hz is not None else None,
                mean_request_itl_cycles=sum(request_intervals)/len(request_intervals) if request_intervals else None,
                asymptotic_no_queue_upper_envelope_per_cycle=envelope,
                feedback_cycles=feedback_cycles,
                stages=[dict(id=s['id'], launches=launches[i], peak_slots=peak[i],
                             output_blocked_item_cycles=blocked[i]) for i,s in enumerate(stages)],
                warmup_scope='User-selected number of completions; stationarity not inferred',
                measured_hardware=False)
