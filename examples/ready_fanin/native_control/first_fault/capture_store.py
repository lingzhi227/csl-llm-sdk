"""Durable immutable bit-preserving NPZ receipts for the finite sink and first-fault capture."""
from pathlib import Path
import hashlib,io,json,os
import numpy as np
SPECS={'first-fault':(178,128),'first-fault-final':(178,128),'initial':(356,4),'origin':(1,3072),'peer':(1,1536),'producers':(136,48),'final-status':(178,8),**{f'progress-{i:02d}':(178,8) for i in range(8)}}
def atomic(path,raw,replace=False):
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if replace:os.replace(tmp,path)
    else:os.link(tmp,path);tmp.unlink()
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def save(root,label,value):
    if label not in SPECS or value.dtype!=np.uint32 or value.size!=SPECS[label][0]*SPECS[label][1]:raise ValueError('Exact uint32 capture shape')
    directory=Path(root)/'evidence';directory.mkdir(exist_ok=True)
    if directory.is_symlink():raise ValueError('Owned evidence directory')
    buf=io.BytesIO();np.savez(buf,words=value.reshape(-1));raw=buf.getvalue()
    paths=list(directory.glob('*.npz'))
    if len(paths)>=15 or len(raw)>131072 or sum(p.stat().st_size for p in paths)+len(raw)>1048576:raise ValueError('Finite raw budget')
    p=directory/(label+'.npz');atomic(p,raw)
    return dict(path=str(p.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),shape=list(SPECS[label]))
def progress(root,receipts,counts):
    raw=(json.dumps(dict(status='incomplete',captures=receipts,counts=counts),indent=2)+'\n').encode()
    if len(raw)>65536:raise ValueError('Progress receipt bound')
    atomic(Path(root)/'capture-progress.json',raw,replace=True)
