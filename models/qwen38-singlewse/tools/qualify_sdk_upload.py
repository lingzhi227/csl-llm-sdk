"""Compare real installed SDK protobuf framing via a local recording stub.

No cluster connection, credentials, compilation or physical job is requested.
"""
import argparse,hashlib,json,tempfile,types
from pathlib import Path
from bounded_client import runtime_class
from cerebras.sdk.client.sdk_appliance_client import SdkRuntime

captures=[]
class RecordingStub:
    def sdk_upload_files(self,requests):
        records=[];files={}
        for message in requests:
            raw=message.data_chunk
            assert message.num_bytes==len(raw)
            key=(message.file_name,message.total_bytes,message.compile_hash,message.mode,message.destination_rel_path)
            files.setdefault(key,bytearray()).extend(raw);records.append(len(raw))
        captures.append((files,records))
        return types.SimpleNamespace(code=0,message='local recording stub')

p=argparse.ArgumentParser();p.add_argument('--single-message',action='store_true');a=p.parse_args()
bounded=runtime_class(single_message_limit=128<<20 if a.single_message else None)
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);payload=bytes(range(256))*8200+b'last';paths=[root/'empty',root/'artifact']
    paths[0].write_bytes(b'');paths[1].write_bytes(payload)
    for cls in [SdkRuntime,bounded]:
        instance=object.__new__(cls)
        instance.stub=lambda:RecordingStub()
        instance._upload_files(str(root),'fixture-compile-id',[str(p) for p in paths],None)
    original,streamed=captures
    assert original[0]==streamed[0]
    if a.single_message:assert max(streamed[1])<=128<<20 and len(streamed[1])==2
    else:assert max(streamed[1])<=1<<20 and len(streamed[1])>=4
    assert sum(streamed[1])==len(payload)
    if a.single_message:
        oversized=root/'oversized'
        with oversized.open('wb') as stream:stream.truncate((128<<20)+1)
        instance=object.__new__(bounded)
        instance.stub=lambda:RecordingStub()
        try:instance._upload_files(str(root),'fixture-compile-id',[str(oversized)],None)
        except ValueError:pass
        else:raise AssertionError('Oversized artifact was admitted')
        assert len(captures)==2,'Oversized artifact reached network stub'
result=dict(passed=True,installed_method_pin=True,protobuf_payloads_identical=True,
            single_message_per_file=a.single_message,
            oversized_rejected_before_transport=a.single_message,
            bounded_chunk_bytes=max(streamed[1]),chunks=len(streamed[1]),physical_jobs=0,
            payload_sha256=hashlib.sha256(payload).hexdigest())
Path('COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
