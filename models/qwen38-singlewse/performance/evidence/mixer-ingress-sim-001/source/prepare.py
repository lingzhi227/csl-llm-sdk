"""Bounded original mixer slices, independent packet and ordered FP32 oracles."""
import hashlib,json
from pathlib import Path
import numpy as np
from weights import OriginalWeights
from reference.mixer_operand import prepare
from reference.mlp_oracle import bf16_float,fp8_float,quantize

plan=json.loads(Path('region.json').read_text());meta=json.loads(Path('tensors.json').read_text())
assert meta['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
pins={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
reader=OriginalWeights(Path('/srv/model-storage/qwen38-singlewse/model'),meta['tensors'],pins)
arrays={};cases=[];origins=[];case=0
for owner in plan['workers']:
 bank=np.zeros(323,np.uint32);setup=np.zeros(40,np.uint16);raw_weights=[];scales=[]
 for mi,(matrix,d,offset) in enumerate(zip(plan['matrices'],owner['descriptors'],[0,65,130,194,258])):
  setup[mi*8:mi*8+8]=[offset,min(d['iterations'],1),d['first_row'],d['row_stride'],d['key_index'],d['key_parts'],d['bf16'],d['rows']]
  rows=d['rows'];column=d['key_index']*128;name=matrix['tensor'];raw=np.zeros((rows,128),np.uint16 if d['bf16'] else np.uint8);scale=np.float32(1)
  if d['iterations']:
   raw=reader.rows(name,d['first_row'],rows)[:,column:column+128].copy()
   if not d['bf16']:scale=bf16_float(reader.rows(name+'_scale_inv',d['first_row']//128,1)[0,d['key_index']])[()]
  values=bf16_float(raw).astype(np.float64) if d['bf16'] else fp8_float(raw)
  raw_weights.append(values);scales.append(scale)
  packed=np.ascontiguousarray(raw.T).reshape(-1).view(np.uint32);bank[offset:offset+len(packed)]=packed
  if not d['bf16']:bank[offset+64]=scale.view(np.uint32)
  origins.append(dict(owner_rank=owner['rank'],matrix=mi,tensor=name,row=d['first_row'],column=column,rows=rows,owned=bool(d['iterations']),original_bank_word_offset=d['base_words'],sample_bank_word_offset=offset))
 arrays['bank_'+str(owner['rank'])]=bank;arrays['setup_'+str(owner['rank'])]=setup
 ids=np.arange(128,dtype=np.uint16)
 for mode in ['mixed','zero_after','changed','replay']:
  bits=(((123+ids%5)<<7)|((ids*31+13)%128)|((ids%2)<<15)).astype(np.uint16)
  if mode=='zero_after':bits=np.where(ids%2,32768,0).astype(np.uint16)
  if mode=='changed':bits=(((117+ids%11)<<7)|((ids*17+7)%128)|(((ids+1)%2)<<15)).astype(np.uint16)
  for parts in (40,48):
   mi=0 if parts==40 else 4;group=owner['descriptors'][mi]['key_index'];invocation=case+1
   packet=prepare(bits,invocation,group,parts);codes,xscales=quantize(bits);expect=np.zeros(10,np.float32)
   for m in (range(4) if parts==40 else [4]):
    if not setup[m*8+1]:continue
    if m in (2,3):weight=raw_weights[m];x=bf16_float(bits).astype(np.float64)
    else:weight=raw_weights[m]/256.;x=fp8_float(codes)/256.
    acc=np.zeros(len(weight),np.float32)
    for k in range(128):acc=(acc.astype(np.float64)+weight[:,k]*x[k]).astype(np.float32)
    if m not in (2,3):acc=np.float32(np.float32(np.float32(acc*65536.)*xscales[0])*scales[m])
    expect[2*m:2*m+len(acc)]=acc
   arrays['input_'+str(case)]=bits.view(np.uint32);arrays['packet_'+str(case)]=packet;arrays['expected_'+str(case)]=expect
   cases.append(dict(case=case,owner_rank=owner['rank'],mode=mode,parts=parts,group=group,invocation=invocation));case+=1
with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
Path('fixture.json').write_text(json.dumps(dict(revision=meta['revision'],original_shards=pins,origins=origins,cases=cases,
 fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),
 criterion='Exact original packet, native codes, candidate/baseline and independent ordered FP32 outputs after zero-sign canonicalization; bank/input/setup retained; zero/change/replay and normal stop. Local arithmetic only, no fabric/full-bank/full-projection/model acceptance.'),indent=2)+'\n')
