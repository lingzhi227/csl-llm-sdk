"""One physical context with a strong owner through stop and failure cleanup."""
import json
import logging
import os
import time
from checkpoint import Restore
from runtime_gate import ROOT, verify, artifact, restore_source
from runtime_prepared import Prepared
from runtime_families import load
from runtime_store import Store
from runtime_io import IO
from runtime_capture import capture
from runtime_timing import Timing, atomic_json


def phase(name, seconds):
    now = time.monotonic()
    atomic_json(ROOT/'runtime-phase.json', dict(phase=name, started=now, deadline=now+seconds))


def main():
    inputs, profiles, bindings = verify('runtime')
    profile, compiled = profiles['runtime'], artifact(inputs)
    host, lowered = [json.loads((ROOT/name).read_bytes()) for name in ('host-plan.json', 'lowered.json')]
    budget = host['baseline_budget' if inputs['mode'] == 'baseline' else 'restored_budget']
    if (profile['max_raw_bytes'], profile['max_raw_files']) != (budget['raw_byte_ceiling'], budget['raw_file_ceiling']):
        raise ValueError('Exact finite mode-specific capture bounds')
    prepared, families = Prepared(inputs['prepared']), load()
    pins = {k: v['pin'] for k, v in inputs['prepared'].items()}
    restore = None
    if inputs['mode'] == 'restore':
        _, baseline_root, _, checkpoint = restore_source(inputs)
        restore = Restore(baseline_root, checkpoint, host, lowered, inputs['artifact_pin']['sha256'], pins)
    from hw00.job_capture import install
    from startup_trace import StartupTrace, sdk_targets
    install(os.environ['HW00_JOB_CAPTURE'])
    logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkRuntime
    from cerebras.sdk.client import sdk_appliance_client as sdk_module
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
    store = Store(ROOT, max_raw_bytes=profile['max_raw_bytes'], max_journal_bytes=profile['max_journal_bytes'],
                  max_files=profile['max_raw_files'], max_events=profile['max_events'])
    runner = io = result = primary = stop_error = None
    entered = normal_stop = False
    try:
        phase('startup', profile['startup_seconds'])
        store.event('runtime_enter')
        with StartupTrace(ROOT, store, sdk_targets(sdk_module), profile):
            runner = SdkRuntime(str(compiled), simulator=False, disable_version_check=True,
                                resource_cpu=profile['worker_cpu_millicores'], resource_mem=profile['worker_memory_bytes'])
            runner.__enter__()
            entered = True
        store.event('runtime_ready')
        phase('capture', profile['capture_seconds'])
        timing = Timing(ROOT, profile, store)
        io = IO(runner, MemcpyDataType, MemcpyOrder, bindings, store, budget, timing.before)
        result = capture(io, prepared, ROOT, host, lowered, families, profile, timing,
                         inputs['artifact_pin']['sha256'], pins, restore)
    except BaseException as exc:
        primary = exc
    finally:
        atomic_json(ROOT/'monitor-deadline.json', dict(active=False, reason='capture_exit_normal_stop_or_failure_cleanup'))
        atomic_json(ROOT/'weight-upload-deadline.json', dict(active=False, reason='capture_exit_normal_stop_or_failure_cleanup'))
        # io owns every outstanding asynchronous buffer until stop or parent reap.
        if entered:
            try:
                phase('stop', profile['normal_stop_seconds'])
                store.event('stop_enter')
                runner.__exit__(None, None, None)
                normal_stop = True
                store.event('stop_exit')
            except BaseException as exc:
                stop_error = exc
        atomic_json(ROOT/'runtime-lifecycle.json', dict(entered=entered, normal_stop=normal_stop,
                    primary_error=None if primary is None else repr(primary),
                    stop_error=None if stop_error is None else repr(stop_error)))
        store.close()
    if primary is not None:
        raise primary
    if stop_error is not None:
        raise stop_error
    if not normal_stop or result is None:
        raise RuntimeError('Complete capture and actual normal physical stop required')
    result.update(normal_stop=True, journal_events=store.sequence, journal_bytes=store.journal_bytes)
    atomic_json(ROOT/'capture.json', result)
    atomic_json(ROOT/'runtime-phase.json', dict(phase='stopped', stopped=True))
    verify('runtime')


if __name__ == '__main__':
    main()
