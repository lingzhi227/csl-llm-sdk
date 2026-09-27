"""Common bounded experiment runtime selection; no host neural computation."""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path

def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def message_limit(profile):
    if profile.get('full_resident',False):return 128<<20
    limit=profile.get('artifact_single_message_limit')
    ceiling=8<<20
    # The first complete 1,262-PE matrix produces a measured 12,170,645-byte
    # archive. Larger transport is admitted only for this exact bounded profile.
    if (profile.get('complete_original_matrix') is True and
        profile.get('application')==[631,2] and profile.get('application_pes')==1262 and
        profile.get('matrix_shape')==[48,5120] and profile.get('fabric_offset')==[123,706]):
        ceiling=16<<20
    if limit is not None and not 0<limit<=ceiling:
        raise ValueError('Component artifact exceeds its bounded geometry profile')
    return limit

@contextmanager
def runtime(physical):
    if physical:
        from job_capture import install
        install('run.jobs')
        from bounded_client import runtime_class
        profile=json.loads(Path('experiment.json').read_text()) if Path('experiment.json').exists() else {}
        limit=message_limit(profile)
        SdkRuntime=runtime_class(single_message_limit=limit)
        from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType,MemcpyOrder
        artifact=json.loads(Path('artifact.json').read_text())
        path=Path(artifact['artifact'])
        if limit is not None and path.stat().st_size>limit:
            raise ValueError('Artifact exceeds bounded one-message upload profile before allocation')
        if file_sha256(path)!=artifact['sha256']:
            raise ValueError('Artifact identity changed')
        with SdkRuntime(str(path),simulator=False,disable_version_check=True,resource_cpu=2000,resource_mem=4<<30) as runner:
            yield runner,MemcpyDataType,MemcpyOrder
    else:
        from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
        runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3))
        runner.load();runner.run()
        try:yield runner,MemcpyDataType,MemcpyOrder
        finally:runner.stop()
