"""Bounded streamed per-program metadata; no full-population Python report list."""
from pathlib import Path
import hashlib,json,os


class Writer:
    def __init__(self,root,profile):
        self.root=Path(root);self.root.mkdir();self.profile=profile
        self.buffer=bytearray();self.in_shard=0;self.count=self.bytes=0
        self.chunks=[];self.digest=hashlib.sha256()

    def append(self,record):
        raw=(json.dumps(record,separators=(',',':'))+'\n').encode()
        assert len(raw)<=self.profile['program_record_bytes']<=self.profile['program_shard_bytes']
        assert self.count<self.profile['maximum_ELFs'] and self.bytes+len(raw)<=self.profile['program_metadata_bytes']
        if self.in_shard>=self.profile['program_shard_records'] or len(self.buffer)+len(raw)>self.profile['program_shard_bytes']:self.flush()
        self.buffer.extend(raw);self.in_shard+=1;self.count+=1;self.bytes+=len(raw);self.digest.update(raw)

    def flush(self):
        if not self.in_shard:return
        path=self.root/f'part-{len(self.chunks):04}.jsonl'
        with path.open('xb') as f:f.write(self.buffer);f.flush();os.fsync(f.fileno())
        path.chmod(0o444)
        self.chunks.append(dict(path=path.name,records=self.in_shard,bytes=len(self.buffer),sha256=hashlib.sha256(self.buffer).hexdigest()))
        self.buffer=bytearray();self.in_shard=0

    def finish(self):
        self.flush();assert self.count>0
        result=dict(format='complete-ELF-program-shards-v1',programs=self.count,bytes=self.bytes,
                    stream_sha256=self.digest.hexdigest(),chunks=self.chunks)
        raw=(json.dumps(result,separators=(',',':'))+'\n').encode();assert len(raw)<=128<<10
        path=self.root/'manifest.json'
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        path.chmod(0o444);fd=os.open(self.root,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return dict(path='program-index/manifest.json',bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),programs=self.count,metadata_bytes=self.bytes)
