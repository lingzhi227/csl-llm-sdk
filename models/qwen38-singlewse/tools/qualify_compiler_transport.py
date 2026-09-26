"""Installed SDK protobuf upload comparison and multi-chunk download checks."""
import hashlib,json,tempfile,types
from pathlib import Path
from bounded_compiler import compiler_class,receive
from cerebras.sdk.client import SdkCompiler

captures=[]
class Stub:
    def sdk_upload_files(self,requests):
        files={};sizes=[]
        for message in requests:
            assert message.num_bytes==len(message.data_chunk) and message.mode=='SDK_COMPILE'
            key=(message.file_name,message.total_bytes,message.compile_hash,message.mode,message.destination_rel_path)
            files.setdefault(key,bytearray()).extend(message.data_chunk);sizes.append(message.num_bytes)
        captures.append((files,sizes));return types.SimpleNamespace(code=0,message='recording stub')

def response(raw,*,name='fixture.tar.gz',total=19,num=None):
    return types.SimpleNamespace(HasField=lambda name:False,data=types.SimpleNamespace(file_name=name,total_bytes=total,num_bytes=len(raw) if num is None else num,data_chunk=raw))

bounded=compiler_class();checks=[]
with tempfile.TemporaryDirectory(dir='.') as temporary:
    base=Path(temporary);payload=bytes(range(256))*8200+b'last';empty=base/'empty';path=base/'payload';empty.write_bytes(b'');path.write_bytes(payload)
    for cls in [SdkCompiler,bounded]:
        instance=object.__new__(cls);instance.stub=lambda:Stub()
        instance._upload_files(str(base),'fixture-id',[str(empty),str(path)])
    assert captures[0][0]==captures[1][0] and max(captures[1][1])<=8<<20 and len(captures[1][1])==2
    checks.append('Original SDK and bounded compile protobuf payloads/metadata match')
    folder=base/'valid';folder.mkdir();parts=[b'hello',b' resident',b' world']
    assert sum(map(len,parts))==20
    got=receive([response(x,total=20) for x in parts],folder,'fixture',max_total=64,max_chunk=16)
    assert Path(got).read_bytes()==b''.join(parts);checks.append('Three download chunks appended with exact bytes')
    cases={
        'truncated':[response(b'short')],
        'oversize':[response(b'a'*20)],
        'wrong_filename':[response(b'x',name='../outside')],
        'inconsistent_total':[response(b'hello'),response(b'rest',total=20)],
        'wrong_num_bytes':[response(b'x',num=2)],
        'chunk_cap':[response(b'a'*19)],
        'empty':[],
    }
    for name,stream in cases.items():
        folder=base/('test_'+name);folder.mkdir()
        try:receive(stream,folder,'fixture',max_total=64,max_chunk=16)
        except (ValueError,AssertionError):pass
        else:raise AssertionError('Accepted malformed '+name)
        assert not (folder/'fixture.tar.gz').exists();checks.append('Rejected '+name)
result=dict(passed=True,physical_jobs=0,installed_upload_and_download_ast_pins=True,checks=checks,
    upload_max_chunk=max(captures[1][1]),upload_payload_sha256=hashlib.sha256(payload).hexdigest())
Path('COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
