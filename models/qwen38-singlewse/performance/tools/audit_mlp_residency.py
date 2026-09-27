"""Independent compact-domain/interval proof for the MLP-anchored residency.

All PE profiles and all auxiliary/state addresses are enumerated. Matrix stream
bijections are proved from disjoint contiguous intervals and checked at every
matrix boundary; this does not read all checkpoint values or run the model.
"""
from collections import Counter
import math
import re


def audit(r,graph,tensors):
    role=bytearray([5])*870000
    def place(x,y,value,previous=5):
        i=y*750+x;assert role[i]==previous;role[i]=value
    for group in range(136):
        x,y=(group%9)*83,40+(group//9)*65
        for p in range(64):
            place(x+41,y+1+p,0)
            for k in range(40):place(x+40-k,y+1+p,1);place(x+42+k,y+1+p,2)
        for j in range(40):place(x+40-j,y,4)
    for c in range(9):
        for k in range(40):place(c*83+40-k,k,4);place(c*83+42+k,k,0)
    for k in range(40):place(0,k,0)
    state_owners=set()
    for layer_index,layer in enumerate(r.gdn_layers):
        for head in range(48):
            number=layer_index*48+head;group,cell=divmod(number,160)
            for ks in range(4):
                for vs in range(4):
                    x=(group%9)*83+42+4*(cell%10)+vs
                    y=40+(group//9)*65+1+4*(cell//10)+ks
                    place(x,y,3,previous=2);address=r.gdn_address(layer,head,ks,vs)
                    assert address['xy']==[x,y] and address['bytes']==4096
                    assert address['rows']==[32*ks,32*ks+32] and address['columns']==[32*vs,32*vs+32]
                    assert (x,y) not in state_owners;state_owners.add((x,y))
    assert role==r.roles and len(state_owners)==36864
    matrix_count=Counter();total_payload=0;actual_max=0;extra=bf_stream=aux=0;helpers=[]
    for i,kind in enumerate(role):
        fp,bf,base,state=int(r.fp8[i]),int(r.bf16[i]),int(r.base[i]),int(r.state[i])
        assert base==(128 if kind==1 else 64 if kind in [2,3] else 0)
        assert state==(4096 if kind==3 else 0) and fp>=base
        if kind==0:assert fp==bf==0
        if kind==4:assert fp==0 and bf==68;helpers.append(i)
        matrix_count['fp8']+=fp;matrix_count['bf16']+=bf
        weight_bytes=fp*260+bf*512;total_payload+=weight_bytes
        assert r.fp8_prefix[i]==extra and r.bf16_prefix[i]==bf_stream and r.aux_prefix[i]==aux
        extra+=fp-base
        if kind!=4:bf_stream+=bf
        free_pages=0 if kind in [0,4] else (35256-weight_bytes-state)//128
        assert free_pages>=0
        used_pages=min(free_pages,max(0,r.used_aux_pages-aux));aux+=free_pages
        actual_max=max(actual_max,weight_bytes+state+128*used_pages)
        assert actual_max<=35256
        bank=r.bank(i);assert bank['matrix_bytes']==weight_bytes and bank['aux_capacity_pages']==free_pages
        assert bank['bf16_base']==fp*260 and bank['fp8_scale_base']==fp*256
        assert bank['gdn_state_offset']==weight_bytes and bank['aux_base']==weight_bytes+state
    assert extra==r.fp8_prefix[-1]==28180480 and bf_stream==r.bf16_prefix[-1]==9630560
    assert aux==r.aux_prefix[-1] and helpers==list(r.helpers) and len(helpers)==5800
    names=list(dict.fromkeys(w for node in graph['nodes'] for w in node['weights']))
    expected_matrix=Counter();used=set();intervals={'other-fp8':[],'bf16':[]};mlp=set();boundary_checks=0
    for name,m in r.matrices.items():
        s=tensors[name];assert s['shape']==m['shape'];rows,cols=s['shape'];count=rows//2*(cols//128)
        assert m['tiles']==count;kind='fp8' if s['dtype']=='F8_E4M3' else 'bf16';expected_matrix[kind]+=count;used.add(name)
        if kind=='fp8':
            scale=name+'_scale_inv';used.add(scale)
            assert tensors[scale]['dtype']=='BF16' and tensors[scale]['shape']==[math.ceil(rows/128),cols//128]
        if '.mlp.' in name:
            match=re.fullmatch(r'model.language_model.layers.(\d+).mlp.(gate|up|down)_proj.weight',name)
            assert match and m['mapping']=='mlp-fixed';layer=int(match[1]);branch=match[2]
            assert (m['layer'],m['branch'])==(layer,branch);mlp.add((layer,branch))
        else:
            assert m['mapping'] in intervals
            length=count-m.get('reserved_prefix',0)
            intervals[m['mapping']].append((m['stream_start'],length))
        samples={0,count-1}
        if name==r.embedding:samples.update([394399,394400])
        for tile in samples:
            a=r.address(name,tile);x,y=a['xy'];bank=r.bank(y*750+x);row,k=divmod(tile,cols//128)
            assert a['rows']==[2*row,2*row+2] and a['columns']==[128*k,128*k+128]
            assert a['byte_offset']>=0 and a['byte_offset']+a['bytes']<=bank['matrix_bytes']
            if '.mlp.' in name:
                if branch=='down':g,p,j=k,row%64,row//64;slot=2*layer+1;xx=40-j
                else:g,p,j=row//64,row%64,k;slot=2*layer if branch=='gate' else layer;xx=40-j if branch=='gate' else 42+j
                assert a['xy']==[83*(g%9)+xx,40+65*(g//9)+1+p] and a['slot']==slot
            if kind=='fp8':
                assert a['original_scale_index']==[row//64,k]
                assert a['scale_byte_offset']==bank['fp8_scale_base']+4*a['slot']
                assert a['scale_byte_offset']+4<=bank['bf16_base']
            boundary_checks+=1
    assert mlp=={(l,b) for l in range(64) for b in ['gate','up','down']}
    assert matrix_count==expected_matrix=={'fp8':95027200,'bf16':10024960}
    for kind,entries in intervals.items():
        cursor=0
        for start,length in sorted(entries):assert start==cursor and length>0;cursor+=length
        assert cursor==({'other-fp8':28180480,'bf16':9630560}[kind])
    # Every embedding-only cohost slot is a distinct original prefix tile.
    for rank,index in enumerate(helpers):
        for slot in [0,67]:
            a=r.address(r.embedding,rank*68+slot)
            assert a['xy']==[index%750,index//750] and a['slot']==slot
            assert a['format']=='row-major-2x128' and a['byte_offset']==slot*512
    for n in r.norms:
        used.add(n['tensor']);assert tensors[n['tensor']]['shape']==[5120]
        for group in range(40):
            a=r.norm_address(n['tensor'],group)
            assert a==dict(xy=[0,group],byte_offset=n['slot']*256,bytes=256,rows=[128*group,128*group+128])
    assert sorted(n['slot'] for n in r.norms)==list(range(129))
    cursor=0;kv=conv=original_aux=0
    for a in r.auxiliary:
        assert a['page_start']==cursor and a['pages']==(a['bytes']+127)//128;cursor+=a['pages']
        if a['kind']=='original-weight':
            used.add(a['id']);assert a['bytes']==tensors[a['id']]['bytes'];original_aux+=a['bytes']
        elif a['id'].startswith('kv:'):kv+=a['bytes'];assert a['shape']==[2,4,96,256]
        else:conv+=a['bytes'];assert a['id'].startswith('conv:') and a['shape']==[3,10240]
    assert cursor==r.used_aux_pages and used==set(names) and len(used)==1251
    assert kv==6291456 and conv==2949120
    previous=None
    for page in range(cursor):
        a=r.auxiliary_address(page);x,y=a['xy'];i=y*750+x;bank=r.bank(i)
        key=(i,a['byte_offset']);assert previous is None or key>previous;previous=key
        assert role[i] not in [0,4] and a['byte_offset']>=bank['aux_base'] and a['byte_offset']+128<=35256
    assert sum(tensors[n]['bytes'] for n in used)==29468003328
    return dict(passed=True,original_tensors=1251,original_bytes=29468003328,matrix_tiles=sum(expected_matrix.values()),
                resident_matrix_bytes_including_scale_copies=total_payload,bank_profiles_enumerated=870000,
                embedding_helpers=5800,embedding_helper_tiles=394400,norm_owners=40,norm_bytes_per_owner=33024,
                gdn_shards_enumerated=len(state_owners),gdn_state_bytes=150994944,kv_state_bytes=kv,conv_state_bytes=conv,
                original_auxiliary_weight_bytes=original_aux,auxiliary_pages_enumerated=cursor,auxiliary_capacity_pages=aux,
                maximum_matrix_state_and_allocated_aux_payload=actual_max,matrix_boundary_addresses_checked=boundary_checks,
                whole_matrix_tile_bijection='Disjoint fixed MLP domains and layer slots; disjoint exact other-matrix stream/PE capacity intervals; disjoint helper embedding prefix. Matrix values are not read by this audit.',
                compiled_all_role_sram_admitted=False,full_model_executable=False,physical=False)
