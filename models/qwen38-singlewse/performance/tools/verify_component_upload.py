"""No-job proof of original SDK framing for a real bounded component artifact.

Run in a staged attempt with the installed SDK and bounded_client.py available.
A recording stub replaces transport; this script never creates a runtime/job.
"""
import hashlib,json,types
from pathlib import Path
from bounded_client import runtime_class
from cerebras.sdk.client.sdk_appliance_client import SdkRuntime
artifact=json.loads(Path('artifact.json').read_text());path=Path(artifact['artifact']);payload=path.read_bytes()
assert 0<len(payload)<=8<<20 and hashlib.sha256(payload).hexdigest()==artifact['sha256']
captures=[]
class RecordingStub:
 def sdk_upload_files(self,requests):
  messages=list(requests);assert all(m.num_bytes==len(m.data_chunk) and m.total_bytes==len(payload) for m in messages)
  assert b''.join(m.data_chunk for m in messages)==payload
  captures.append([m.SerializeToString() for m in messages]);return types.SimpleNamespace(code=0,message='local recording stub')
for cls in [SdkRuntime,runtime_class(),runtime_class(single_message_limit=8<<20)]:
 instance=object.__new__(cls);instance.stub=lambda:RecordingStub();instance._upload_files(str(path.parent),'component-proof',[str(path)],None)
assert captures[0]==captures[2] and len(captures[0])==1
record=dict(passed=True,physical_jobs=0,artifact_bytes=len(payload),artifact_sha256=artifact['sha256'],original_sdk_messages=len(captures[0]),prior_bounded_messages=len(captures[1]),candidate_messages=len(captures[2]),candidate_protobuf_identical_to_original=True,cap_bytes=8<<20,
 scope='Recording-stub protobuf identity for this exact artifact. No network upload or server-side truncation proof.')
with Path('upload-framing.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
