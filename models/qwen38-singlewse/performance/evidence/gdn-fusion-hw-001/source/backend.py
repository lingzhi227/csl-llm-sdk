"""Common bounded experiment runtime selection; no host neural computation."""
from contextlib import contextmanager
import hashlib,json
from pathlib import Path

def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def complete_mlp_profile(profile):
    return (profile.get('complete_original_mlp') is True and
            profile.get('stage')=='layer_00' and profile.get('application')==[78,146] and
            profile.get('application_pes')==11388 and profile.get('mlp_shape')==[5120,17408,5120] and
            profile.get('fabric_offset')==[67,1] and
            profile.get('revision')=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a')

def complete_mixer_profile(profile):
    return (profile.get('complete_original_mixer') is True and
            profile.get('stage')=='layer_00' and profile.get('application')==[78,146] and
            profile.get('application_pes')==11388 and profile.get('fabric_offset')==[67,1] and
            profile.get('mixer_shapes')==[[10240,5120],[6144,5120],[48,5120],[48,5120],[5120,6144]] and
            profile.get('revision')=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a')

def gdn_columns_profile(profile):
    return (profile.get('original_gdn_columns') is True and profile.get('full_model') is False
            and profile.get('application')==[251,6] and profile.get('application_pes')==1506
            and profile.get('gdn_shape')==[48,128,128] and profile.get('gdn_workers')==753
            and profile.get('fabric_offset')==[4,1]
            and profile.get('revision')=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a')

def gdn_fusion_profile(profile):
    return (profile.get('original_frontend_gdn_fusion') is True and profile.get('full_model') is False
            and profile.get('application')==[160,5] and profile.get('application_pes')==800
            and profile.get('frontend_shape')==[16,3,128]
            and profile.get('gdn_shape')==[48,128,128] and profile.get('gdn_workers')==753
            and profile.get('host_injected_recurrent_results') is False
            and profile.get('fabric_offset')==[4,1]
            and profile.get('revision')=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a')

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
    if complete_mlp_profile(profile):ceiling=64<<20
    # Complete routed mixer018 contains 7,714 ELF images; a bounded no-output
    # tar/gzip census measures 101,664,618B. Keep this admission distinct from MLP.
    if complete_mixer_profile(profile):ceiling=128<<20
    if gdn_columns_profile(profile) or gdn_fusion_profile(profile):ceiling=16<<20
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
