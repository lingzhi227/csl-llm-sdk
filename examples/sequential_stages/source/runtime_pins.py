"""Bounded immutable file pins without retaining large artifacts in memory."""
from pathlib import Path
import hashlib,json,stat
from stage_contract import require


def stamp(path):
    value=Path(path).lstat()
    return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode)


def relative(name):
    path=Path(name)
    require(isinstance(name,str) and name and not path.is_absolute() and '..' not in path.parts and
            str(path)==name,'Canonical relative immutable path')
    return path


def pinned(path,pin,limit,*,retain=False):
    path=Path(path);before=stamp(path)
    require(stat.S_ISREG(before[5]) and not before[5]&0o222 and not path.is_symlink() and
            type(pin['bytes']) is int and 0<=pin['bytes']==before[2]<=limit,'Bounded readonly pinned input: '+str(path))
    digest=hashlib.sha256();chunks=[];total=0
    with path.open('rb') as stream:
        while True:
            raw=stream.read(1<<20)
            if not raw:break
            total+=len(raw);require(total<=pin['bytes'],'Input cannot grow during verification')
            digest.update(raw)
            if retain:chunks.append(raw)
    require(total==pin['bytes'] and digest.hexdigest()==pin['sha256'] and stamp(path)==before,'Complete stable immutable pin: '+str(path))
    return b''.join(chunks) if retain else before


def document(path,pin,limit):return json.loads(pinned(path,pin,limit,retain=True))


def sealed_document(path,limit):
    """Read a small readonly root record whose content supplies its own pin."""
    path=Path(path);before=stamp(path)
    require(stat.S_ISREG(before[5]) and not before[5]&0o222 and 0<before[2]<=limit,'Bounded sealed root metadata')
    raw=path.read_bytes();require(stamp(path)==before,'Stable sealed root record')
    return json.loads(raw),dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def programs(root,index_pin):
    """Exhaust the exact inspector stream, checking each complete shard."""
    root=Path(root);path=root/relative(index_pin['path'])
    require(str(relative(index_pin['path']))=='program-index/manifest.json','Exact program index location')
    index=document(path,index_pin,128<<10)
    require(index['format']=='complete-ELF-program-shards-v1' and
            0<index['programs']==index_pin['programs']<=32768 and
            0<index['bytes']==index_pin['metadata_bytes']<=128<<20 and 0<len(index['chunks'])<=512,'Bounded complete ELF program stream')
    digest=hashlib.sha256();total=count=0
    for number,chunk in enumerate(index['chunks']):
        require(chunk['path']==f'part-{number:04}.jsonl' and 0<chunk['records']<=512,'Ordered unique bounded program shards')
        raw=pinned(path.parent/chunk['path'],chunk,4<<20,retain=True)
        digest.update(raw);total+=len(raw);lines=raw.splitlines(keepends=True)
        require(len(lines)==chunk['records'],'Complete program shard record count')
        for line in lines:
            require(0<len(line)<=1<<20 and line.endswith(b'\n'),'Bounded complete program JSONL record')
            count+=1;require(count<=index['programs'],'No excess program record')
            yield json.loads(line)
    require(count==index['programs'] and total==index['bytes'] and digest.hexdigest()==index['stream_sha256'],
            'Every complete actual program shard exhausted and hashed')
