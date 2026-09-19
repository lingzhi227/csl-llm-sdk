"""Bind a separately accepted physical artifact before allocating the finite fixture."""
import logging,os,time
from source_gate import ROOT,PROFILE,verify
from compiled_binding import preflight
from hw00.job_capture import install
from hw00.store import atomic_json
from capture import capture
from validate_capture import captured_result

def main():
    verify();artifact,binding=preflight();install(os.environ['HW00_JOB_CAPTURE']);logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkRuntime
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType,MemcpyOrder
    lifecycle=[]
    def mark(kind):
        lifecycle.append(dict(kind=kind,monotonic_seconds=time.monotonic()));atomic_json(ROOT/'runtime-lifecycle.json',lifecycle)
    class Runtime:
        def __init__(self):
            self.closed=False;self.stop_attempted=False
            self.runtime=SdkRuntime(str(artifact),simulator=False,disable_version_check=True,resource_cpu=PROFILE['worker_cpu_millicores'],resource_mem=PROFILE['worker_memory_bytes'])
            self.runtime.__enter__()
        def __getattr__(self,name):return getattr(self.runtime,name)
        def stop(self):
            if self.closed:return
            if self.stop_attempted:raise RuntimeError('Failed context exit already attempted; supervisor owns cleanup')
            self.stop_attempted=True;mark('context_exit_start');self.runtime.__exit__(None,None,None);self.closed=True;mark('context_exit_returned')
    runner=None
    try:
        mark('context_enter_start');runner=Runtime();mark('context_enter_returned')
        atomic_json(ROOT/'monitor-deadline.json',dict(active=True,deadline_monotonic=time.monotonic()+PROFILE['lifecycle_seconds'],scope='capture and normal context exit after entry'))
        result=capture(ROOT/'cases/lifecycle',lambda:runner,MemcpyDataType,MemcpyOrder)
    finally:
        if runner is not None and not runner.stop_attempted:runner.stop()
        if runner is not None and runner.closed:atomic_json(ROOT/'monitor-deadline.json',dict(active=False,reason='normal_exit_returned'))
    if not result['normal_stop'] or not all(v['passed'] for v in result['checks'].values()):raise ValueError('Complete exact records and normal exit required')
    captured_result()
    atomic_json(ROOT/'physical.json',dict(simulator=False,artifact_sha256=binding['artifact']['sha256'],runtime_context_exited=runner.closed,application=[3,1],packets=9,application_words=200,rounds=1,reset=False,complete_original_neural_epochs=0,scope='Same three-PE bidirectional200-word physical comparison; full rows, causal markers, banks, stability and both first-packet witnesses required'))
    verify()
if __name__=='__main__':main()
