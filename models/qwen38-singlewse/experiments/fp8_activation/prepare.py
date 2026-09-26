"""Freeze PyTorch FP8 reference for finite encodings, midpoint ties and groups."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
codes=torch.arange(256,dtype=torch.uint8)
decoded=codes.view(torch.float8_e4m3fn).float().numpy()
positive=decoded[:127]
mid=(positive[:-1]+positive[1:])*np.float32(.5)
edges=np.concatenate([mid,np.nextafter(mid,np.float32(-np.inf)),np.nextafter(mid,np.float32(np.inf))])
all_values=np.concatenate([decoded[np.isfinite(decoded)],edges,-edges]).astype(np.float32)
all_values=np.pad(all_values,(0,(-len(all_values))%128))
groups=all_values.reshape(-1,128)
extras=np.stack([np.zeros(128,np.float32),np.sin(np.arange(128)*.31).astype(np.float32),
                 np.linspace(-1e-14,1e-14,128,dtype=np.float32),np.linspace(-448,448,128,dtype=np.float32)])
groups=np.concatenate([groups,extras])
x=torch.from_numpy(groups)
scales=x.abs().amax(-1,keepdim=True).clamp_min(1e-10)*np.float32(1/448)
q=(x/scales).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
direct=x.clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
with Path('fixture.npz').open('xb') as stream:np.savez(stream,vectors=groups,scales=scales.numpy(),codes=q,direct=direct)
meta=dict(fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),groups=len(groups),
          scope='All 254 finite encodings; positive and negative midpoint ties and neighboring FP32 values; zero and tiny groups',
          criterion='Exact FP8 output bytes; FP32 scales within 1 ULP; frozen before device outputs',
          backend='vLLM group-128 convention, use_ue8m0=False; independent PyTorch float8_e4m3fn conversion',torch_version=torch.__version__)
Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
