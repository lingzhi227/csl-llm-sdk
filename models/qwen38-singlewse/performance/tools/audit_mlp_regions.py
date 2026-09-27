"""Independent streaming audit of original tensor ownership and known routes.

Uses bounded byte arrays, not a million-entry Python PE/route dictionary. Missing
stream families and missing compiled code remain explicit admission failures.
"""
from array import array
from collections import Counter
from itertools import chain


def audit(document, tiles, native_edges):
    d=document;w,h=d['application'];cols=d['spec']['columns'];regions=d['regions']
    assert len(regions)==136 and (w,h)==(83*cols,40+65*((135//cols)+1))
    assert w<=750 and h<=1160
    ownership=bytearray(w*h);coverage=[bytearray(348160) for _ in range(3)]
    weights=d['projection_tensor_names'];counts=Counter();matrix_bytes=0
    for t in tiles:
        which=weights.index(t['tensor']);r0,r1=t['rows'];k0,k1=t['columns'];x,y=t['xy']
        m,k=([17408,5120] if which<2 else [5120,17408])
        assert 0<=r0<r1<=m and r1-r0==2 and r0%2==0
        assert 0<=k0<k1<=k and k1-k0==128 and k0%128==0
        assert t['scale_tensor']==t['tensor']+'_scale_inv' and t['scale_index']==[r0//128,k0//128]
        idx=(r0//2)*(k//128)+k0//128
        assert not coverage[which][idx];coverage[which][idx]=1
        if which<2:
            g,pair,q=r0//128,r0%128//2,k0//128
            expected=(83*(g%cols)+(40-q if which==0 else 42+q),40+65*(g//cols)+1+pair)
            assert t['phase']=='gate-up'
            assert ownership[y*w+x]==0;ownership[y*w+x]=1
        else:
            g,pair,q=k0//128,r0%128//2,r0//128
            expected=(83*(g%cols)+40-q,40+65*(g//cols)+1+pair)
            assert t['phase']=='down'
            # Iterator may be reordered, so independently verify coordinate;
            # reuse population is checked after complete tile coverage.
        assert (x,y)==expected and 0<=x<w and 0<=y<h
        counts[which]+=1;matrix_bytes+=256
    assert all(all(a) for a in coverage) and [counts[i] for i in range(3)]==[348160]*3
    assert matrix_bytes==267386880==d['ownership']['original_matrix_bytes']
    def actor(x,y):
        assert 0<=x<w and 0<=y<h and not ownership[y*w+x]
        ownership[y*w+x]=2
    for g,region in enumerate(regions):
        ox,oy=83*(g%cols),40+65*(g//cols)
        assert region['id']==g and region['origin']==[ox,oy] and region['extent']==[83,65]
        assert region['intermediate_rows']==[128*g,128*g+128]
        for p in range(64):actor(ox+41,oy+p+1)
        for j in range(40):actor(ox+40-j,oy)
    for bx in range(cols):
        for j in range(40):
            actor(bx*83+40-j,j);actor(bx*83+42+j,j)
    for j in range(40):actor(0,j)
    assert sum(v>0 for v in ownership)==d['ownership']['distinct_active_pes']

    # A byte holds RX direction+1. Different sources at one PE/color are illegal
    # even when they would use different output links. Include endpoints.
    rx=bytearray(w*h*18)
    loads=array('I',[0])*(w*h*4)
    kinds=Counter();stream_count=0
    # Reconstruct the ordered K tree independently of contraction.tree().
    parents={};levels={};sides={}
    def split(start,count,parent=-1,level=0,side=0):
        parents[start]=parent;levels[start]=level;sides[start]=side
        if count>1:
            left=count//2;split(start+1,left,start,level+1,0)
            if count-1-left:split(start+1+left,count-1-left,start,level+1,1)
    split(0,40)
    seen_native=bytearray(136*64*2*40)
    seen_input=set()
    dirs={(1,0):0,(-1,0):1,(0,1):2,(0,-1):3}
    for s in chain(d['input_distribution'],d['down_edges'],native_edges):
        stream_count+=1
        a=s['source'];b=s.get('destination',s.get('end'));color=s['color'];words=s['words']
        assert 2<=color<=19 and 0<words<=d['spec']['maximum_return_stream_words']
        parts=s['id'].split(':')
        if parts[0] in ['native','fused-pair']:
            g,p,branch=int(parts[1]),int(parts[2]),parts[3];assert branch in ['gate','up']
            bx,by=g%cols,g//cols;origin_x=83*bx;yy=40+65*by+1+p
            rank=int(parts[4]) if parts[0]=='native' else 0
            assert 0<=g<136 and 0<=p<64 and 0<=rank<40
            index=((g*64+p)*2+(branch=='up'))*40+rank
            assert not seen_native[index];seen_native[index]=1
            xx=lambda r:origin_x+(40-r if branch=='gate' else 42+r)
            assert a==[xx(rank),yy]
            if rank:
                assert b==[xx(parents[rank]),yy] and color==4+2*(levels[rank]-1)+sides[rank] and words==3
            else:
                assert b==[origin_x+41,yy] and color==(16 if branch=='gate' else 17) and words==1
        if parts[0]=='normalized-packet':
            k=int(parts[1]);assert 0<=k<40 and color==2 and words==65
            assert a==[0,k] and b==[83*(cols-1)+42+k,k]
            assert s['consumers']==[[c*83+x,k] for c in range(cols) for x in [40-k,42+k]]
            assert s['id'] not in seen_input;seen_input.add(s['id'])
        if parts[0]=='input':
            c,branch,k=int(parts[1]),parts[2],int(parts[3]);groups=list(range(c,136,cols))
            assert 0<=c<cols and 0<=k<40 and branch in ['gate','up']
            xx=83*c+(40-k if branch=='gate' else 42+k)
            assert a==[xx,k] and b==[xx,40+65*(groups[-1]//cols)+64]
            assert color==3 and words==65 and s['groups']==groups and s['consumers']==64*len(groups)
            assert s['id'] not in seen_input;seen_input.add(s['id'])
        assert (a[0]==b[0]) != (a[1]==b[1])
        dx=(b[0]>a[0])-(b[0]<a[0]);dy=(b[1]>a[1])-(b[1]<a[1]);direction=dirs[dx,dy]
        length=abs(a[0]-b[0])+abs(a[1]-b[1]);kinds[s['kind']]+=length*words
        for step in range(length+1):
            x,y=a[0]+step*dx,a[1]+step*dy;assert 0<=x<w and 0<=y<h
            key=(y*w+x)*18+color-2
            # No same-color stream merging is assumed. A turn is a real actor.
            assert rx[key]==0, ('PE/color collision',s['id'],x,y,color)
            rx[key]=5 if step==0 else (direction^1)+1
            if step<length:loads[(y*w+x)*4+direction]+=words
    assert max(loads)==128
    assert all(seen_native) and len(seen_input)==40+2*40*cols
    assert dict(sorted(kinds.items()))==d['cost']['known_word_hops_by_family']
    # Independent path endpoints and tensor extents; counts alone are insufficient.
    expected_ids=set()
    for s in d['down_edges']:
        prefix,g,j=s['id'].split(':');g=int(g);j=int(j)
        assert s['words']==128 and s['output_rows']==[128*j,128*j+128]
        assert s['chunk_words']==d['spec']['down_chunk_words']
        if prefix=='down-v':
            expected_ids.add((prefix,g,j));bx,by=g%cols,g//cols
            assert s['source']==[bx*83+40-j,40+65*by]
            assert s['destination']==([bx*83+40-j,40+65*(by-1)] if by else [bx*83+40-j,j])
            assert s['color']==13+by%2
        else:
            assert prefix=='down-h';expected_ids.add((prefix,g,j))
            assert s['source']==[g*83+40-j,j]
            assert s['destination']==([(g-1)*83+40-j,j] if g else [0,j])
            assert s['color']==17+g%2
    assert expected_ids=={('down-v',g,j) for g in range(136) for j in range(40)}|{('down-h',g,j) for g in range(cols) for j in range(40)}
    leaves=[]
    def visit(e):
        if isinstance(e,int):leaves.append(e);return 0
        assert len(e)==2;return 1+max(visit(e[0]),visit(e[1]))
    depth=visit(d['down_fp32_reduction_expression'])
    assert sorted(leaves)==list(range(136))
    assert not d['executable'] and not d['all_routes_lowered'] and not d['compiled_sram_qualified']
    return dict(passed=True, original_matrix_tiles=sum(counts.values()), original_matrix_bytes=matrix_bytes,
                distinct_active_pes=sum(v>0 for v in ownership), known_streams=stream_count,
                known_route_entries=sum(v>0 for v in rx), known_directed_links=sum(v>0 for v in loads),
                known_directed_link_max_words=max(loads), down_reduction_leaves=len(leaves),
                down_fp32_binary_depth=depth, known_word_hops_by_family=dict(kinds),
                all_streams_audited=False, executable=False, physical=False, full_model=False,
                scope='Every original FP8 matrix tile, replicated scale index and active actor coordinate; input multicast, every gate/up contraction and adjacent packed activation operand, and inter-region down streams. Missing families, leases, slot packing, numerical order and compiled SRAM are not admitted.')
