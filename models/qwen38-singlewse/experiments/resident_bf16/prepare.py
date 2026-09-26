"""Original first/last BF16 vocabulary tiles and frozen independent FP64 oracle."""
import hashlib,json,struct
from pathlib import Path
import numpy as np

def bits(x):
    v=np.asarray(x,np.float32).copy().view(np.uint32)
    return ((v+np.uint32(0x7fff)+((v>>16)&1))>>16).astype(np.uint16)
def expand(v):return (v.astype(np.uint32)<<16).view(np.float32)

source=Path('/srv/model-storage/qwen38-singlewse/model/outside.safetensors')
pin='ddff1d6665a2b39f2612fce0ef955e2436724c565bfbcbc127c7ffd078b698ff'
h=hashlib.sha256()
with source.open('rb') as stream:
    for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
assert h.hexdigest()==pin
with source.open('rb') as stream:
    header_size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(header_size))
    def tile(name,first_row,k):
        spec=header[name];assert spec['dtype']=='BF16' and spec['shape']==[248320,5120]
        result=np.zeros((128,140),np.uint16)
        for r in range(min(140,248320-first_row)):
            stream.seek(8+header_size+spec['data_offsets'][0]+((first_row+r)*5120+k*128)*2)
            raw=stream.read(256);assert len(raw)==256
            result[:,r]=np.frombuffer(raw,dtype='<u2')
        return result.reshape(-1)
    tiles=[];weights=[]
    for row in range(4):
        embedding=row<2;matrix=row+1 if embedding else 3
        name='model.language_model.embed_tokens.weight' if embedding else 'lm_head.weight'
        first_row=0 if row%2==0 else 248220
        for k in range(2 if embedding else 3):
            tiles.append(dict(x=[0,4,1,4][row]+k,y=row,matrix=matrix,k=k,tensor=name,first_row=first_row,valid_rows=min(140,248320-first_row)))
            weights.append(tile(name,first_row,k))
weights=np.array(weights)
index=np.arange(384,dtype=np.float64)
inputs=bits(np.stack([np.sin(index*.17)*8,((index%19)-9)/16,np.zeros(384)]))
controls=np.array([[7,99],[139,0],[0,31]],np.uint32)
embedding=np.zeros((3,512),np.float32)
exact=np.zeros((3,280),np.float64);absolute=np.zeros_like(exact)
for i,t in enumerate(tiles):
    w=expand(weights[i].reshape(128,140)).astype(np.float64)
    if t['matrix']<3:
        for call in range(3):
            offset=(t['matrix']-1)*256+t['k']*128
            embedding[call,offset:offset+128]=w[:,controls[call,t['matrix']-1]]
    else:
        values=expand(inputs[:,t['k']*128:(t['k']+1)*128]).astype(np.float64)
        offset=(t['y']-2)*140
        exact[:,offset:offset+140]+=values@w
        absolute[:,offset:offset+140]+=np.abs(values)@np.abs(w)
gamma=400*2.0**-24/(1-400*2.0**-24)
bounds=gamma*absolute+np.finfo(np.float32).tiny
np.savez('fixture.npz',weights=weights,inputs=inputs,controls=controls,embedding=embedding,exact=exact,bounds=bounds)
meta=dict(scope='Original BF16 first/last vocabulary embedding tiles and three-K-block head submatrix; device masked argmax',
    model='Qwen/Qwen3.8-27B-FP8',revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',shard_sha256=pin,
    tiles=tiles,application_pes=45,command_frames_per_call=13,full_model=False,full_head=False,
    criterion='Embedding bit identity; head FP64 gamma400 bound; argmax over valid BF16 rounded outputs with lowest-ID tie; winner inside frozen rounded-oracle interval candidates; all45 traces and persistent original weights',
    fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest())
Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)
