"""Independent FP64 RMSNorm oracle on every captured model layer input.

The bound is one BF16 ULP, fixed before the new device result. FP64 is rounded
directly to the BF16 lattice to avoid a FP32 midpoint double-rounding artifact.
"""
import hashlib,json
from pathlib import Path
import numpy as np

root=Path.cwd();model=Path('/srv/qwen38-singlewse-hardware/model')
captured=Path('/srv/qwen38-singlewse-hardware/resident-generation-hw-002/request-00')
generation=json.loads((captured/'generation.json').read_text());positions=generation['processed_positions']
tokens=generation['prompt_ids']+generation['generated_ids'][:-1];assert len(tokens)==positions==66
meta=json.loads((root/'tensors.json').read_text());assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
pins=json.loads((model/'COMPLETE.json').read_text());assert pins['revision']==meta['revision'] and len(pins['files'])==74
path=captured/'layers.bf16';receipt=json.loads((captured/'layers-capture.json').read_text())
assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['sha256']
layers=np.memmap(path,dtype='<u2',mode='r',shape=(positions,64,5120))
stamps={}
def tensor(name,row=None):
 v=meta['tensors'][name];assert v['dtype']=='BF16';p=model/v['shard'];assert not p.is_symlink()
 st=p.stat();stamps[p]=(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
 offset=v['data_start']+v['data_offsets'][0];shape=v['shape']
 if row is not None:offset+=row*shape[1]*2;shape=[shape[1]]
 return np.array(np.memmap(p,mode='r',dtype='<u2',offset=offset,shape=tuple(shape)),copy=True)
def expand(bits):return (bits.astype(np.uint32)<<16).view(np.float32)
inputs=np.empty((65,positions,5120),np.uint16);gain=np.empty((65,5120),np.uint16);expected=np.empty_like(inputs)
for layer in range(65):
 name=f'model.language_model.layers.{layer}.input_layernorm.weight' if layer<64 else 'model.language_model.norm.weight'
 gain[layer]=tensor(name)
 for position in range(positions):
  bits=tensor('model.language_model.embed_tokens.weight',tokens[position]) if layer==0 else layers[position,layer-1]
  inputs[layer,position]=bits;x=expand(bits).astype(np.float64)
  factor=(np.float32(1)+expand(gain[layer])).astype(np.float64)
  gold=(x/np.sqrt(np.mean(x*x)+1e-6))*factor
  quantum=np.ldexp(np.ones_like(gold),np.maximum(np.frexp(np.abs(gold))[1]-8,-133))
  rounded=np.copysign(np.rint(np.abs(gold)/quantum)*quantum,gold)
  expected[layer,position]=(rounded.astype(np.float32).view(np.uint32)>>16).astype(np.uint16)
for p,before in stamps.items():
 st=p.stat();assert before==(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
np.savez('fixture.npz',inputs=inputs,gain=gain,fp64=expected)
result=dict(cases=65*positions,positions=positions,hidden_size=5120,physical=False,criteria_fixed_before_candidate_outputs=True,
 max_bf16_ulp_from_fp64=1,require_no_more_total_differences_than_baseline=True,
 scope='Every captured pre-layer input and final-normalization input; original gains. This does not accept the full model.',
 candidate_source='resident-generation-hw-002',capture_sha256=receipt['sha256'],revision=meta['revision'],
 fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest())
Path('fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
