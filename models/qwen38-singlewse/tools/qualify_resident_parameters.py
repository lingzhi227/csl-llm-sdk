"""Verify new small-tensor reads against original headers and actor coordinates."""
import argparse,hashlib,json,struct,time
from pathlib import Path
import numpy as np
from weights import OriginalWeights
from resident_plan import load

p=argparse.ArgumentParser();p.add_argument('--resident',type=Path,required=True);args=p.parse_args()
root=args.resident
model=Path('/srv/qwen38-singlewse-hardware/model')
complete=json.loads((model/'COMPLETE.json').read_text());assert len(complete['files'])==74
plan=load(root);hub=json.loads((root/'hub.json').read_text())
pins={s['rfilename']:s['lfs']['sha256'] for s in hub['siblings'] if s['rfilename'].endswith('.safetensors')}
reader=OriginalWeights(model,plan['tensors'],pins);headers={};count=0;total=0;started=time.monotonic()
for name,assignment in plan['atlas']['assignments'].items():
    if assignment.get('role')!='other':continue
    value=reader.small_tensor(name);metadata=plan['tensors'][name];shard=metadata['shard']
    with (model/shard).open('rb') as stream:
        if shard not in headers:
            length=struct.unpack('<Q',stream.read(8))[0];assert length<=4<<20
            headers[shard]=(length+8,json.loads(stream.read(length)))
        start,header=headers[shard];entry=header[name]
        stream.seek(start+entry['data_offsets'][0]);raw=stream.read(entry['data_offsets'][1]-entry['data_offsets'][0])
    assert value.shape==tuple(entry['shape']) and value.tobytes()==raw and value.dtype==np.dtype('<u2')
    count+=1;total+=len(raw)
assert count==353
for layer in range(64):
    if layer%4==3:continue
    groups=[g for g in plan['roles']['gdn'] if g['layer']==layer]
    assert sorted(g['value_head'] for g in groups)==list(range(48))
    for group in groups:
        head=group['value_head'];assert group['conv_channel_starts']==[(head//3)*128,2048+(head//3)*128,4096+head*128]
        index=(layer-layer//4)*48+head
        xy=[744,154+index] if index<555 else [720+3*((index-555)%9),709+(index-555)//9]
        assert group['controller']==xy
assert len(plan['roles']['gdn'])==2304 and len(plan['roles']['attention'])==64
for group in plan['roles']['attention']:
    x,y=group['controller'];assert x==744 and 90<=y<154
    assert (y-90)//4==group['layer']//4
    prefix=f"model.language_model.layers.{group['layer']}.self_attn."
    for suffix in ['q_norm.weight','k_norm.weight']:assert reader.small_tensor(prefix+suffix).shape==(256,)
receipt=dict(passed=True,physical_jobs=0,small_original_tensors=count,original_bytes=total,gdn_groups=2304,attention_groups=64,
    full_resident_model=False,source_manifest={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['weights.py','resident_loader.py','resident_plan.py']},
    role_sha256=plan['role_sha256'],config_sha256=plan['config_sha256'],seconds=time.monotonic()-started,
    scope='Exact independent raw-header comparison of every new small-tensor read and every GDN parameter slice/actor coordinate; earlier FP8/BF16 packing and actual ELF role audit are separate gates')
Path('COMPLETE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
