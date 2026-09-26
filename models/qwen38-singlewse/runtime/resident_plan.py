"""Exact role/configuration inventory shared by CSL layout and original loader."""
import hashlib,json
from pathlib import Path
import numpy as np

def load(root):
    root=Path(root)
    atlas=json.loads((root/'placement-resident-96-bf16-140.json').read_text())
    roles=json.loads((root/'role-plan-96.json').read_text())
    matrices=json.loads((root/'device-matrices-bf16-140.json').read_text())
    tensors=json.loads((root/'tensors.json').read_text())['tensors']
    assert hashlib.sha256((root/'placement-resident-96-bf16-140.json').read_bytes()).hexdigest()==roles['weight_atlas_sha256']==matrices['placement_sha256']
    width,height=atlas['application'];assert (width,height)==(750,1160)
    role=np.zeros((height,width),np.uint8);phase=np.zeros_like(role);first=np.zeros_like(role);last=np.zeros_like(role)
    config=np.zeros((height,width,8),np.uint32)
    config[:,:,0]=np.arange(width,dtype=np.uint32)[None,:]+(np.arange(height,dtype=np.uint32)[:,None]<<10)
    occupied=np.zeros_like(role)
    def claim(x,y,n,value):
        assert 0<=x<x+n<=width and 0<=y<height and not occupied[y,x:x+n].any()
        occupied[y,x:x+n]=1;role[y,x:x+n]=value
    for name,a in atlas['assignments'].items():
        if 'strips' in a:
            m,k=a['shape'];rows=a['tile'][0];n=k//128;matrix=matrices['matrix_ids'][name]
            for ordinal,x,y in a['strips']:
                fp8=a['role']=='fp8_matrix';claim(x,y,n,1 if fp8 else 2)
                config[y,x:x+n,1]=matrix;config[y,x:x+n,2]=np.arange(n,dtype=np.uint32)
                config[y,x:x+n,3]=ordinal;config[y,x:x+n,4]=min(rows,m-ordinal*rows)
                config[y,x:x+n,7]=17408 if fp8 else 17920
                phase[y,x:x+n]=(ordinal*272)%128 if fp8 else 0
                first[y,x]=1;last[y,x+n-1]=1
        elif a.get('role')=='other':
            nbytes=tensors[name]['bytes'];norm=name in roles['normalization']
            for i,(x,y) in enumerate(a['coordinates']):
                claim(x,y,1,4 if norm else 3);config[y,x,7]=min(32768,nbytes-i*32768)//2
            assert len(a['coordinates'])==(nbytes+32767)//32768
    assert occupied.sum()==849313
    for x,y in atlas['bus']['coordinates']:claim(x,y,1,14 if [x,y]==roles['controller'] else 0)
    for group in roles['gdn']:
        x,y=group['controller'];claim(x,y,3,0);role[y,x:x+3]=[6,7,8];config[y,x:x+3,5]=x+(y<<10)
    for group in roles['attention']:
        x,y=group['controller'];claim(x,y,5,0);role[y,x:x+5]=[9,10,11,12,13];config[y,x:x+5,5]=x+(y<<10)
    for bank in roles['activation_banks']:claim(*bank['coordinate'],1,5)
    for xy in roles['validation']['coordinates']:claim(*xy,1,5)
    assert (occupied==0).sum()==8347
    assert np.count_nonzero(role==1)==703104 and np.count_nonzero(role==2)==145760
    assert np.count_nonzero(role==5)==3199 and np.count_nonzero(role==14)==1
    return dict(atlas=atlas,roles=roles,matrices=matrices,tensors=tensors,role=role,phase=phase,first=first,last=last,config=config,
        role_sha256=hashlib.sha256(role.tobytes()).hexdigest(),config_sha256=hashlib.sha256(config.astype('<u4').tobytes()).hexdigest())
