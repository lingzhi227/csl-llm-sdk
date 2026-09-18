"""One full resident layer3 physical capture, with all payload checks up front."""
import hashlib,json,logging,os
from pathlib import Path
from source_gate import ROOT,verify,PROFILE
from hw00.job_capture import install
from hw00.store import atomic_json
from full_capture import load_prepared,execute,require,hash_file
from audit_full import reference_inputs

def main():
    verify();inputs=json.loads((ROOT/'inputs.json').read_bytes())
    artifact=Path(inputs['artifact'])
    require(hash_file(artifact)==inputs['artifact_sha256'],'Accepted physical artifact')
    check=Path(inputs['postcheck'])
    for name,spec in inputs['postcheck_files'].items():
        path=check/name;require(path.stat().st_size==spec['bytes'] and hash_file(path)==spec['sha256'],'Accepted full artifact check identity')
    require(json.loads((check/'checked.json').read_bytes())['passed'],'Full artifact postcheck passed')
    receipt,hidden,stamps=load_prepared(inputs['prepared'],inputs['preparation_sha256'])
    reference_inputs(inputs)  # Reuse/hash accepted arrays, never rerun a forward.
    # Every534MBoriginal payload byte and full physical artifact is bound before
    # constructing the runtime. All subsequent math verification occurs offline.
    install(os.environ['HW00_JOB_CAPTURE']);logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkRuntime
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType,MemcpyOrder
    class Runtime:
        def __init__(self):
            self.closed=False
            self.runtime=SdkRuntime(str(artifact),simulator=False,disable_version_check=True,
                                    resource_cpu=2000,resource_mem=4<<30)
            self.runtime.__enter__()
        def __getattr__(self,name):return getattr(self.runtime,name)
        def stop(self):
            if not self.closed:self.runtime.__exit__(None,None,None);self.closed=True
    runner=None
    try:
        runner=Runtime();result=execute(runner,MemcpyDataType,MemcpyOrder,ROOT,inputs['prepared'],receipt,hidden,stamps)
    finally:
        if runner is not None:runner.stop()
    require(result['status']=='captured' and result['normal_stop'],'Complete capture and runtime context exit')
    atomic_json(ROOT/'physical.json',dict(simulator=False,artifact_sha256=inputs['artifact_sha256'],runtime_context_exited=runner.closed,
                                        original_layer=3,dimensions=[5120,17408,5120],mathematical_audit_passed=False,full_model=False))
    verify()

if __name__=='__main__':main()
