"""Original embedding payload plus ordered independent partial-join fixtures."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
metadata=json.loads(Path('tensors.json').read_text());assert metadata['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
pin={'outside.safetensors':'ddff1d6665a2b39f2612fce0ef955e2436724c565bfbcbc127c7ffd078b698ff'}
model=Path('/srv/model-storage/qwen38-singlewse/model')
reader=OriginalWeights(model,metadata['tensors'],pin)
name='model.language_model.embed_tokens.weight';original=reader.rows(name,0,4)
bank=np.zeros(8814,np.uint32);tiles=[]
for i in range(68):
 row,k=divmod(i,40);tile=np.ascontiguousarray(original[row*2:row*2+2,k*128:k*128+128])
 bank[i*128:(i+1)*128]=tile.reshape(-1).view('<u4');tiles.append(dict(slot=i,rows=[row*2,row*2+2],columns=[k*128,k*128+128]))
rng=np.random.default_rng(512017408);payload=rng.normal(0,2,(3,3,128)).astype(np.float32)
payload[1]=0;payload[0,0,:4]=[1e20,1,-1e20,-1];payload[0,1,:4]=[-1e20,1,1e20,-1];payload[0,2,:4]=[1,-2,-1,2]
packet=np.stack([np.arange(65,dtype=np.uint32)+0x1000+case*0x10000 for case in range(3)])
partial=(payload[:,0]+payload[:,1]).astype(np.float32);expected=(partial+payload[:,2]).astype(np.float32)
with Path('fixture.npz').open('xb') as f:np.savez(f,bank=bank,payload=payload,packet=packet,partial=partial,expected=expected)
d=dict(fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),model=metadata['repository'],revision=metadata['revision'],
       tensor=name,shard_sha256=pin,tiles=tiles,original_payload_bytes=34816,padding_bytes=440,actor_allocated_bank_bytes=35256,
       shape=[3,3],helpers=[[1,1],[1,0]],chunks=16,chunk_words=8,epochs=3,
       scope='Actual resident embedding lookup plus relay/credited two-stage FP32 join. Sources supply synthetic partial vectors; no original down projection or complete MLP execution/timing claim.',
       criterion='All original bank/padding words retained; embedding lookups exact; input relay exact; every intermediate/final ordered FP32 value and callback counter exact; three warm epochs; normal stop.',physical=False,full_model=False)
Path('fixture.json').write_text(json.dumps(d,indent=2)+'\n')
