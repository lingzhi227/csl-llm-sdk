"""Common bounded experiment runtime selection; no host neural computation."""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path

@contextmanager
def runtime(physical):
    if physical:
        from job_capture import install
        install('run.jobs')
        from cerebras.sdk.client import SdkRuntime
        from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType,MemcpyOrder
        artifact=json.loads(Path('artifact.json').read_text())
        path=Path(artifact['artifact'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
            raise ValueError('Artifact identity changed')
        with SdkRuntime(str(path),simulator=False,disable_version_check=True,resource_cpu=2000,resource_mem=4<<30) as runner:
            yield runner,MemcpyDataType,MemcpyOrder
    else:
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
        runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3))
        runner.load();runner.run()
        try:yield runner,MemcpyDataType,MemcpyOrder
        finally:runner.stop()
