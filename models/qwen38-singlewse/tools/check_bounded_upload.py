"""Exercise actual bounded reads, multi-file framing and mutation rejection."""
import io,json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from bounded_client import upload_chunks

with tempfile.TemporaryDirectory() as folder:
    root=Path(folder);payload=bytes(range(256))*39+b'final';empty=root/'empty';data=root/'data'
    empty.write_bytes(b'');data.write_bytes(payload)
    reads=[];real_open=Path.open
    class Guarded:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def fileno(self):return self.stream.fileno()
        def read(self,n=-1):
            assert 0<n<=1024, 'Unbounded read';reads.append(n);return self.stream.read(n)
    def guarded(path,*args,**kwargs):return Guarded(real_open(path,*args,**kwargs))
    with patch.object(Path,'open',guarded):chunks=list(upload_chunks(root,[empty,data],1024))
    assert chunks[0]==('empty',b'',0)
    assert b''.join(block for name,block,total in chunks if name=='data')==payload
    assert all(total==len(payload) and name=='data' for name,block,total in chunks[1:])
    assert len(chunks)>3 and max(reads)==1024
    iterator=upload_chunks(root,[data],1024);next(iterator)
    with real_open(data,'ab') as stream:stream.write(b'changed')
    try:list(iterator)
    except ValueError:pass
    else:raise AssertionError('Mutation was not rejected')
print(json.dumps(dict(passed=True,bounded_read_bytes=1024,multiple_chunks=True,empty_file=True,mutation_rejected=True)))
