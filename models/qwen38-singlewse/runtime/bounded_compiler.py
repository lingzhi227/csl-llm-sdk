"""Task-local compile upload and correctly framed multi-chunk artifact download."""
import ast,hashlib,inspect,json,os,textwrap
from pathlib import Path
from bounded_client import upload_chunks,UPLOAD_AST_SHA256
DOWNLOAD_AST='e5d2c81349ca9bd687a949adc299ad373303952762e9baeabb8f55b17135a0c9'

def receive(iterator,root,name,*,max_total=32<<30,max_chunk=128<<20):
    root=Path(root);expected=name+'.tar.gz';partial=root/(expected+'.partial');final=root/expected
    assert root.is_dir() and Path(name).name==name and not final.exists()
    count=0;total=None;largest=0
    with partial.open('xb') as output:
        for response in iterator:
            if response.HasField('status'):raise ValueError('SDK artifact download returned status: '+str(response.status))
            data=response.data
            if data.file_name!=expected:raise ValueError('Unexpected artifact file name')
            if total is None:total=data.total_bytes
            if not 0<total<=max_total or data.total_bytes!=total:raise ValueError('Artifact total framing')
            block=data.data_chunk;size=len(block)
            if not 0<size<=max_chunk or size!=data.num_bytes or count+size>total:raise ValueError('Artifact chunk framing/limit')
            output.write(block);count+=size;largest=max(largest,size)
        if count!=total:raise ValueError('Incomplete artifact download')
        output.flush();os.fsync(output.fileno())
    partial.replace(final)
    (root/'download-framing.json').write_text(json.dumps(dict(bytes=count,max_server_chunk_bytes=largest,
        multi_chunk_append=True,server_chunk_limit=max_chunk,scope='Server chooses response chunk sizes; local worker AS/RSS limits remain enforced'))+'\n')
    return str(final)

def compiler_class():
    from cerebras.sdk.client import SdkCompiler
    from cerebras.sdk.client import sdk_appliance_client as sdk
    for method,pin in [(SdkCompiler._upload_files,UPLOAD_AST_SHA256),(SdkCompiler.download_artifact,DOWNLOAD_AST)]:
        source=textwrap.dedent(inspect.getsource(method))
        if hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()!=pin:raise ValueError('Installed compiler protocol changed')
    class BoundedSdkCompiler(SdkCompiler):
        def _upload_files(self,app_path,compile_hash,files,destination_rel_path=None):
            # Observed cluster COMPILE handling keeps only the final message for
            # a repeated filename. Sources are small enough for one bounded
            # message each; never split one source across requests here.
            if any(Path(path).stat().st_size>8<<20 for path in files):raise ValueError('Single-message source bound')
            def requests():
                for name,block,total in upload_chunks(app_path,files,8<<20):
                    yield sdk.sdk_appliance_pb2.SdkArtifactsArgs(file_name=name,data_chunk=block,num_bytes=len(block),
                        total_bytes=total,compile_hash=compile_hash,mode='SDK_COMPILE',destination_rel_path=destination_rel_path or '')
            response=self.stub().sdk_upload_files(requests())
            if response.code in [sdk.sdk_common_pb2.StatusCode.SDK_RT_INTERNAL_ERROR,sdk.sdk_common_pb2.StatusCode.SDK_RT_FILE_ERROR]:raise RuntimeError(response.message)
            return response
        def download_artifact(self,name,out_path):
            responses=self.stub().sdk_download_files(sdk.sdk_appliance_pb2.SdkArtifactDownloadArgs(file_name=name))
            return receive(responses,out_path,name)
    return BoundedSdkCompiler
