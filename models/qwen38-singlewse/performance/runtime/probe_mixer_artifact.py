"""Read one cached compiler response envelope, without compiling or saving payload."""
import json, logging
from pathlib import Path
from job_capture import install
from source_gate import verify


def main():
    verify(); install('compile.jobs'); logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkCompiler
    from cerebras.sdk.client import sdk_appliance_client as sdk
    name='cs_f036c8b388ae59260b1c85d5ae21efc65f3b3ecddf0e88d80b5ee1c3e39ce80b'
    result=dict(artifact=name,origin='layer-mixer-hw-001',compile_requested=False,
                runtime_requested=False,payload_saved=False)
    with SdkCompiler(disable_version_check=True,resource_cpu=2000,resource_mem=4<<30) as compiler:
        responses=compiler.stub().sdk_download_files(
            sdk.sdk_appliance_pb2.SdkArtifactDownloadArgs(file_name=name))
        try:
            response=next(responses)
            if response.HasField('status'):
                result['status']=str(response.status)[:2048]
            else:
                data=response.data
                result.update(file_name=data.file_name,total_bytes=data.total_bytes,
                              num_bytes=data.num_bytes,chunk_bytes=len(data.data_chunk))
        finally:
            result['stream_cancelled']=responses.cancel()
        Path('artifact-envelope.json').write_text(json.dumps(result,indent=2)+'\n')
    verify()
    print(json.dumps(result))


if __name__=='__main__':main()
