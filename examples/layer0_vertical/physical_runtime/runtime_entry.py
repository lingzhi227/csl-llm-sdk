"""One exact physical context; always attempt normal stop after context entry."""
import hashlib,json,logging,os,time
from runtime_gate import ROOT,verify,artifact
from runtime_plan import build
from runtime_prepared import Prepared
from runtime_store import Store
from runtime_io import IO
from runtime_capture import Timing,capture,atomic_json

def phase(name,seconds):
    now=time.monotonic();atomic_json(ROOT/'runtime-phase.json',dict(phase=name,started=now,deadline=now+seconds))

def main():
    inputs,profiles,bindings=verify('runtime');profile=profiles['runtime'];compiled=artifact(inputs)
    plan,transport,host,typed=[json.loads((ROOT/n).read_bytes()) for n in ('plan.json','transport-map.json','host-map.json','EXPORTS.json')]
    copies=build(plan,transport,host,typed)
    extra=[*copies['state_copies'],*copies['transport_copies'],*copies['diagnostic_copies'],copies['observer']]
    raw=(json.dumps(extra,sort_keys=True,separators=(',',':'))+'\n').encode()
    if bindings['inputs']['extra_copy_geometry']!=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()):raise ValueError('All runtime geometries equal actual bound records')
    if copies['budget']!=bindings['runtime_copy_budget']:raise ValueError('Frozen finite original copy schedule')
    prepared=Prepared(inputs['prepared_root'],inputs['prepared_pin'])
    # Job capture stays byte-exact. Startup tracing adds only fixed segment
    # timestamps; both are packaged and pinned in runtime-manifest.
    from hw00.job_capture import install
    from startup_trace import StartupTrace,sdk_targets
    install(os.environ['HW00_JOB_CAPTURE']);logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkRuntime
    from cerebras.sdk.client import sdk_appliance_client as sdk_module
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType,MemcpyOrder
    store=Store(ROOT,max_raw_bytes=profile['max_raw_bytes'],max_journal_bytes=profile['max_journal_bytes'],
                max_files=profile['max_raw_files'],max_events=profile['max_events'])
    runner=None;io=None;entered=False;normal_stop=False;result=None;primary=None;stop_error=None
    try:
        phase('startup',profile['startup_seconds']);store.event('runtime_enter')
        with StartupTrace(ROOT,store,sdk_targets(sdk_module),profile):
            runner=SdkRuntime(str(compiled),simulator=False,disable_version_check=True,
                resource_cpu=profile['worker_cpu_millicores'],resource_mem=profile['worker_memory_bytes'])
            runner.__enter__();entered=True
        store.event('runtime_ready');phase('capture',profile['capture_seconds'])
        timing=Timing(ROOT,profile,store);io=IO(runner,MemcpyDataType,MemcpyOrder,bindings,store,copies['budget'],timing.before)
        result=capture(io,prepared,ROOT,plan,transport,copies,profile,timing)
    except BaseException as exc:primary=exc
    finally:
        atomic_json(ROOT/'monitor-deadline.json',dict(active=False,reason='capture_exit_normal_stop_or_failure_cleanup'))
        atomic_json(ROOT/'weight-upload-deadline.json',dict(active=False,reason='capture_exit_normal_stop_or_failure_cleanup'))
        # io strongly owns weight_session, Upload and every uncompleted buffer
        # until normal physical stop below, or until parent termination/reap.
        if entered:
            try:
                phase('stop',profile['normal_stop_seconds']);store.event('stop_enter')
                runner.__exit__(None,None,None);normal_stop=True;store.event('stop_exit')
            except BaseException as exc:stop_error=exc
        atomic_json(ROOT/'runtime-lifecycle.json',dict(entered=entered,normal_stop=normal_stop,
            primary_error=None if primary is None else repr(primary),stop_error=None if stop_error is None else repr(stop_error)))
        store.close()
    if primary is not None:raise primary
    if stop_error is not None:raise stop_error
    if not normal_stop or result is None:raise RuntimeError('Complete capture plus normal physical stop required')
    result.update(normal_stop=True,journal_events=store.sequence,journal_bytes=store.journal_bytes)
    atomic_json(ROOT/'capture.json',result);atomic_json(ROOT/'runtime-phase.json',dict(phase='stopped',stopped=True));verify('runtime')

if __name__=='__main__':main()
