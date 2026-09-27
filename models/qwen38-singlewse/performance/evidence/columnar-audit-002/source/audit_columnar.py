"""Independent full-address, input-identity and physical route audit of columnar overlay."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np


def need(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root):
    from spatial.columnar import lower_input_routes
    root = Path(root); started = time.monotonic()
    base = json.loads((root/'base-atlas.json').read_text()); plan = json.loads((root/'columnar.json').read_text())
    need(sha(root/'base-atlas.json') == plan['base_atlas_sha256'], 'Base identity')
    need((plan['model'], plan['revision']) == (base['model'], base['revision']), 'Model identity')
    for key, expected in plan['preserved_base_fields_sha256'].items():
        need(hashlib.sha256(json.dumps(base[key], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == expected, 'Changed state/actor/value metadata')
    n = 860880; bank = np.arange(n, dtype=np.int64); row = bank//750
    x = np.where(row%2 == 0, bank%750, 749-bank%750); bank_rows = np.asarray(base['geometry']['bank_rows'])
    y = bank_rows[row]; wide_mask = (row<1146)&(x>0)&(x<749)
    # Independent coordinate-to-bank enumeration of the narrower row-major ring.
    wr, wc = np.divmod(np.arange(857208, dtype=np.int64), 748)
    wx = np.where(wr%2 == 0, wc+1, 748-wc)
    wide = wr*750 + np.where(wr%2 == 0, wx, 749-wx)
    need(np.array_equal(np.sort(wide), bank[wide_mask]), 'Full wide-ring geometry')
    excluded = np.zeros(n, bool)
    for rectangle in [g['state_rectangle'] for g in base['gdn']] + [base['state_class_spare_rectangle']]:
        left, top, width, height = rectangle
        excluded |= (x>=left)&(x<left+width)&(y>=top)&(y<top+height)
    eligible = bank[~excluded]; need(len(eligible)==824000, 'BF16 state exclusion')
    maps = {'fp8-main':bank, 'fp8-wide':wide, 'bf16':eligible}
    original = base['matrices']; matrices = plan['matrices']; classes = plan['classes']
    need(len(matrices)==498 and [m['id'] for m in matrices]==list(range(498)), 'Complete matrices')
    for a,b in zip(original, matrices):
        for key in ['id','tensor','kind','shape','row_tiles','k_blocks','tiles','mode']:
            need(a[key]==b[key], 'Original matrix identity/extent')
        expected = 'bf16' if b['kind']=='bf16' else 'fp8-wide' if b['k_blocks']==136 else 'fp8-main'
        need(b['storage_class']==expected, 'Matrix class')
    reservations = {}; visited = {}; padding = {}
    for key,c in classes.items():
        mapping = maps[key]; size = len(mapping); phase = c['phase']; span = c['stream_span']
        need(c['pes']==size and len(np.unique(mapping))==size, 'Class bank bijection')
        q, rem = divmod(span,size); counts = np.full(size,q,dtype=np.int16)
        counts[(np.arange(rem)+phase)%size]+=1
        physical = np.zeros(n,np.int16);physical[mapping]=counts;reservations[key]=physical
        occupied = np.zeros((q+int(rem>0),size),np.uint16); end=0;real=0;pad=0
        for m in [m for m in matrices if m['storage_class']==key]:
            k=m['k_blocks'];start=(end+k-1)//k*k
            need(start==m['stream_start'] and m['padding_before']==start-end, 'Independent stream prefix')
            need(size%k==phase%k==0 and m['tiles']==m['shape'][0]//2*k, 'Complete K/row groups')
            offset=0
            for segment in m['segments']:
                at=(start+offset)%size;take=min(m['tiles']-offset,size-at)
                need(segment==dict(class_start=at,count=take,slot=(start+offset)//size,tile_start=offset,row_start=offset//k), 'Exact dispatch descriptor')
                need(at%k==take%k==0, 'Partial group')
                for j in range(0,take,131072):
                    ids=np.arange(at+j,at+min(j+131072,take),dtype=np.int64);owner=(ids+phase)%size
                    tensor_tile=segment['slot']*size+ids-start
                    need(np.array_equal(tensor_tile,np.arange(offset+j,offset+min(j+131072,take))), 'Tensor tile address')
                    need(not np.any(occupied[segment['slot'],owner]), 'Resident address alias')
                    occupied[segment['slot'],owner]=m['id']+1
                offset+=take
            need(offset==m['tiles'], 'Complete matrix dispatch')
            pad+=start-end;end=start+m['tiles'];real+=m['tiles']
        need((end,real,pad)==(span,c['real_tiles'],c['padding_tiles']), 'Class conservation')
        need(np.count_nonzero(occupied)==real, 'Every original tile once')
        visited[key]=real;padding[key]=pad
        del occupied
    fp=reservations['fp8-main']+reservations['fp8-wide'];bf=reservations['bf16'];state=excluded.astype(np.int32)*4096
    # Wide slots begin after all main reservations on this PE, including padding.
    need(np.all(fp<=111) and np.all(bf<=13), 'Original maximum resident capacities')
    payload=fp.astype(np.int32)*260+bf.astype(np.int32)*512+state
    need(np.max(payload)<=35256, 'Original maximum bank data bytes exceeded')
    need(int((fp.astype(np.int64)*260+bf.astype(np.int64)*512).sum())==plan['metrics']['resident_matrix_bytes'], 'Payload conservation')
    profiles=[]
    for p in plan['input_profiles']:
        k=p['k_blocks'];lanes=len(p['vertical_colors']);sources=p['sources'];groups=p['horizontal_groups']
        need(lanes=={40:2,48:4,136:1}[k], 'Input channel domain')
        blocks=[b for g in groups for b in g['packet_blocks']]
        need(sorted(blocks)==list(range(k)), 'Complete original input blocks once at row producers')
        pair_table=np.full((lanes,750,2),-1,np.int32);root_rows=np.full((lanes,750),-1,np.int32);claimed=set()
        for s in sources:
            lane=s['lane'];column=s['column'];group=groups[s['actor_row']]
            need(0<=lane<lanes and pair_table[lane,column,0]==-1, 'Unique column source')
            start=s['horizontal_window']['offset_words'];length=s['horizontal_window']['pass_words'];period=s['horizontal_window']['period_words']
            need(start%65==0 and length==130 and period==65*len(group['packet_blocks']), 'Horizontal filter window')
            need(group['packet_blocks'][start//65:start//65+2]==s['blocks'], 'Selected horizontal pair')
            need(s['vertical_color']==p['vertical_colors'][lane], 'Vertical source color')
            actor=(base['geometry']['actor_rows'][s['actor_row']],column)
            need(actor not in claimed, 'Actor needs more than one simultaneous ingress buffer/filter')
            claimed.add(actor);pair_table[lane,column]=s['blocks'];root_rows[lane,column]=actor[0]
        selected=wide_mask if k==136 else np.ones(n,bool)
        bb=bank[selected];rr=row[selected];xx=x[selected];yy=y[selected]
        rank=rr*748+np.where(rr%2==0,xx-1,748-xx) if k==136 else bb
        expected=rank%k;lane=rr%lanes;packets=pair_table[lane,xx]
        matches=packets==expected[:,None]
        need(np.all(matches.sum(axis=1)==1), 'Every bank receives exact original K block')
        need(np.all(root_rows[lane,xx]>=0), 'Missing actor source')
        colors=[p['horizontal_color'],*p['vertical_colors'],*p['reduction_colors']]
        need(len(colors)==len(set(colors)) and all(2<=c<=20 for c in colors), 'Input/reduction router color overlap')
        words=lower_input_routes(base,p);rx=words&7;tx=words>>3
        need(words.shape==(lanes+1,1160,750) and np.all((rx<=5)&(tx<=31)), 'Dense abstract route shape/ports')
        # Verify every neighbor link in both directions, independently of route generator.
        for plane in range(lanes+1):
            r=rx[plane];t=tx[plane];active=(r!=0);need(np.all((t>0)|(~active)), 'Active dead end')
            north=(t&2)!=0;south=(t&4)!=0;west=(t&8)!=0;east=(t&16)!=0
            need(not north[0].any() and not south[-1].any() and not west[:,0].any() and not east[:,-1].any(), 'Leaves physical boundary')
            need(np.array_equal(north[1:],r[:-1]==3) and np.array_equal(south[:-1],r[1:]==2)
                 and np.array_equal(west[:,1:],r[:,:-1]==5) and np.array_equal(east[:,:-1],r[:,1:]==4), 'Neighbor continuity or unowned input')
            injection=set(zip(*np.where(r==1)));consumers=set(zip(*np.where((t&1)!=0)))
            if plane==0:
                expected_injection={(base['geometry']['actor_rows'][g['actor_row']],0) for g in groups}
                expected_consumers=claimed
            else:
                ll=plane-1
                expected_injection={(int(root_rows[ll,col]),col) for col in range(750) if root_rows[ll,col]>=0}
                take=lane==ll;expected_consumers=set(zip(yy[take].tolist(),xx[take].tolist()))
            need(injection==expected_injection and consumers==expected_consumers, 'Exact source/consumer endpoints')
        target=root/('input-k%d.npy'%k)
        with target.open('xb') as f:np.save(f,words,allow_pickle=False)
        profiles.append(dict(k_blocks=k,input_planes=lanes+1,bank_consumers=len(bb),actor_ingress_pes=len(claimed),
                             router_entries_checked=int(words.size),max_horizontal_payload_words=p['max_horizontal_stream_words'],
                             max_column_payload_words=p['max_column_stream_words'],
                             source_buffer_bytes=520,bank_input_buffer_bytes=260,
                             sources_and_original_blocks_exact=True,all_neighbor_links_checked=True,
                             artifact=dict(name=target.name,bytes=target.stat().st_size,sha256=sha(target))))
    census=Counter(zip(fp.tolist(),bf.tolist(),state.tolist()))
    return dict(schema='wse-paired-column-audit-v1',passed=True,physical=False,weights_read=False,
                plan_sha256=sha(root/'columnar.json'),base_atlas_sha256=sha(root/'base-atlas.json'),
                original_matrices=len(matrices),original_tiles_checked=visited,padding_tiles=padding,
                maximum_bank_data_bytes=int(payload.max()),maximum_fp8_slots=int(fp.max()),maximum_bf16_slots=int(bf.max()),
                resident_matrix_bytes=plan['metrics']['resident_matrix_bytes'],
                bank_profiles=[dict(fp8_slots=f,bf16_slots=b,state_bytes=s,pes=num) for (f,b,s),num in sorted(census.items())],
                input_profiles=profiles,duration_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                compiled_sram_admitted=False,counter_filter_backend_qualified=False,full_model_executable=False,
                full_model_speed_target_achieved=False,
                scope='Complete original storage addressing and all FP8 input source/consumer identities and neighbor routes. Filter windows are abstract; upstream value/quant delivery, BF16 inputs, result scatter, route transitions and numerical runtime remain unqualified.')
