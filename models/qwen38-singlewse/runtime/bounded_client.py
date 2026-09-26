"""Task-local bounded artifact upload; never edits the shared SDK installation.

Preserves the installed upload protocol, but reads one bounded block instead
of constructing an entire artifact in memory before chunking it. The inherited
method AST is pinned so an SDK protocol change requires a fresh audit.
"""
import ast,hashlib,inspect,os,textwrap
from pathlib import Path

UPLOAD_AST_SHA256='89954b9e98a215a8bb8dcbfc07cd9786799698909b88ddf3a2c4a8a44f6ecb22'

def upload_chunks(app_path,files,chunk_bytes):
    if not 0<chunk_bytes<=128<<20:raise ValueError('Upload chunk bound')
    for filename in files:
        path=Path(filename)
        with path.open('rb') as stream:
            before=os.fstat(stream.fileno())
            total=before.st_size;sent=0
            if total==0:yield os.path.relpath(path,app_path),b'',0
            while sent<total:
                block=stream.read(min(chunk_bytes,total-sent))
                if not block:raise ValueError('Artifact shortened during upload')
                sent+=len(block)
                yield os.path.relpath(path,app_path),block,total
            after=os.fstat(stream.fileno())
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('Artifact changed during upload')
            if stream.read(1):raise ValueError('Artifact grew during upload')

def runtime_class(single_message_limit=None):
    from cerebras.sdk.client import SdkRuntime
    from cerebras.sdk.client import sdk_appliance_client as sdk
    source=textwrap.dedent(inspect.getsource(SdkRuntime._upload_files))
    if hashlib.sha256(ast.dump(ast.parse(source)).encode()).hexdigest()!=UPLOAD_AST_SHA256:
        raise ValueError('Installed SDK upload method changed; protocol re-audit required')
    if single_message_limit is not None and not 0<single_message_limit<=128<<20:
        raise ValueError('Whole-artifact single-message ceiling')
    class BoundedSdkRuntime(SdkRuntime):
        def _upload_files(self,app_path,compile_hash,files,destination_rel_path=None):
            chunk_bytes=min(1<<20,sdk.MAX_MESSAGE_LENGTH)
            if single_message_limit is not None:
                # Full COMPILE exposed server truncation for repeated source
                # filenames. Use the original one-message EXECUTE framing for
                # a strictly bounded artifact, without assuming append support.
                if any(Path(path).stat().st_size>single_message_limit for path in files):
                    raise ValueError('Compiled artifact exceeds one-message admission')
                chunk_bytes=single_message_limit
            def requests():
                for name,block,total in upload_chunks(app_path,files,chunk_bytes):
                    yield sdk.sdk_appliance_pb2.SdkArtifactsArgs(file_name=name,data_chunk=block,
                        num_bytes=len(block),total_bytes=total,compile_hash=compile_hash,
                        mode='SDK_EXECUTE',destination_rel_path=destination_rel_path or '')
            response=self.stub().sdk_upload_files(requests())
            self._check_sdk_response(response,'Bounded artifact upload')
            return response
    return BoundedSdkRuntime
