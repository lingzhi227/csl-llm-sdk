"""Small interleaved transport fixture from the accepted full original matrix."""
import hashlib,json
from pathlib import Path
import numpy as np

def decode(raw):
    b=raw.astype(np.uint32);e=(b&127)>>3;m=b&7
    positive=np.where(e==0,m*2.0**-9,(1+m/8)*np.exp2(e.astype(np.int32)-7))
    return np.where(b&128,-positive,positive)

source=Path('/srv/qwen38-singlewse-hardware/fp8-matrix-hw-003/fixture.npz')
pin='9ab7b881f556758bcc5dacfd7b3e33f98ea06e03c453fe6c086600881b3ddf08'
h=hashlib.sha256()
with source.open('rb') as src:
    for block in iter(lambda:src.read(1<<20),b''):h.update(block)
assert h.hexdigest()==pin
with np.load(source,allow_pickle=False) as src:
    original_weights=src['weights'];original_scales=src['scales'];packets=src['packets'];vectors=src['vectors']
tiles=[];weights=[];scales=[]
exact=np.zeros((3,1088),np.float64);absolute=np.zeros_like(exact)
for row in range(4):
    matrix=1 if row<2 else 2;width=2 if row<2 else 3;first=[0,4,1,4][row]
    for k in range(width):
        tiles.append(dict(x=first+k,y=row,matrix=matrix,k=k,original_row_tile=row))
        w=original_weights[row,k];s=original_scales[row,k];weights.append(w);scales.append(s)
        decoded=decode(w.copy().view(np.uint8).reshape(128,272).T)
        q=decode(packets[:,k,:32].copy().view(np.uint8).reshape(3,128))
        factor=packets[:,k,32].copy().view(np.float32).astype(np.float64)[:,None]*s[(row*272%128+np.arange(272))//128]
        exact[:,row*272:(row+1)*272]+=(q@decoded.T)*factor
        absolute[:,row*272:(row+1)*272]+=(np.abs(q)@np.abs(decoded.T))*factor
bound=(300*2.0**-24/(1-300*2.0**-24))*absolute+np.finfo(np.float32).tiny
np.savez('fixture.npz',weights=np.array(weights),scales=np.array(scales),
    inputs=np.concatenate((vectors[:,:256],vectors[:,:384]),axis=1),exact=exact,bounds=bound)
meta=dict(scope='Four interleaved original FP8 strips; two logical submatrices, device-controlled compute/return',
    original_fixture_sha256=pin,tiles=tiles,application_pes=45,command_frames_per_call=12,
    fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),
    criterion='Frozen FP64 gamma300 product bound; all45 frame/call counters and forwarding counts; exact weight/scale retention',
    full_matrix=False,full_model=False)
Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
