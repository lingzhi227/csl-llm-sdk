"""Derived context8 origin-publication limits, shared by CSL and host binding.

Proof and original compiled-source pins: OBSERVER-CAPACITY-PROOF001.md.
These bounds count changed() calls, not host polls or measured averages.
The one-error allowance requires first-error-only publication in both families.
"""
import hashlib
from stage_contract import CONTEXT_CAPACITY, require

CHUNKS_BY_PHASE = (5120 // 128, 6144 // 128, 5120 // 128, 17408 // 128)
CHUNKS = sum(CHUNKS_BY_PHASE)
ATTENTION_READY = 1 + 24 + 1 + 136 + 1
LINEAR_READY = 1 + 48 + 1 + 136 + 1
LINEAR_PRODUCERS = 450
LINEAR_RECORDS = 752
ROW_FRAGMENTS = (64 + 22) // 23

ATTENTION_CALLS = dict(need_chunk=CHUNKS, chunk_received=CHUNKS,
    request_sent=CHUNKS, ready=ATTENTION_READY, root_done=96,
    injected=4, compute=1, finish=1)
LINEAR_WAKES = dict(compute=1, need_chunk=CHUNKS, chunk_received=CHUNKS,
    injected=4, sent=LINEAR_RECORDS + CHUNKS,
    received=LINEAR_PRODUCERS + LINEAR_RECORDS + ROW_FRAGMENTS*CHUNKS + LINEAR_READY)
PER_SERIAL = {'attention': sum(ATTENTION_CALLS.values()),
              'linear': sum(LINEAR_WAKES.values()) + 1}  # finish outside service
FIRST_STICKY_ERROR = 1
LEGACY_PUBLICATIONS = {'attention': 4096, 'linear': 16384}
ORIGINAL_OBSERVER_SHA = {
    'attention': '9ea75a76263700344a3a67ad2f04409e87ecb7055c7b60f043231fbd5351da02',
    'linear': 'cea9243bcb0f0c033fa8f3b6fac5a39d7a6f259928353019be64c2e15e0752bf'}


def publications(family):
    require(CONTEXT_CAPACITY == 8 and family in PER_SERIAL, 'Reviewed context8 observer family')
    return CONTEXT_CAPACITY * PER_SERIAL[family] + FIRST_STICKY_ERROR


def wire_limit(family):
    return 2 * publications(family)


def contract():
    return dict(schema=1, context_capacity=CONTEXT_CAPACITY, chunks_by_phase=list(CHUNKS_BY_PHASE),
        attention_changed_calls=ATTENTION_CALLS, linear_service_wakes=LINEAR_WAKES,
        per_serial=PER_SERIAL, first_sticky_error=FIRST_STICKY_ERROR,
        publications={f:publications(f) for f in PER_SERIAL},
        wire_limits={f:wire_limit(f) for f in PER_SERIAL},
        legacy_publications=LEGACY_PUBLICATIONS, original_observer_sha256=ORIGINAL_OBSERVER_SHA)


def device_observer(text, family):
    require(hashlib.sha256(text.encode()).hexdigest() == ORIGINAL_OBSERVER_SHA[family],
            'Exact accepted origin observer source before capacity adaptation')
    old = 'sequence<' + str(LEGACY_PUBLICATIONS[family])
    require(text.count(old) == 1, 'One original device publication bound')
    return text.replace(old, 'sequence<' + str(publications(family)), 1)


def first_linear_error(text):
    old = 'if(error==0){error=code;state[15]=code;}if(comptime origin){if(initialized){observer.changed();}}'
    new = 'if(error==0){error=code;state[15]=code;if(comptime origin){if(initialized){observer.changed();}}}'
    require(text.count(old) == 1, 'Exact original sticky-error publication site')
    return text.replace(old, new, 1)
