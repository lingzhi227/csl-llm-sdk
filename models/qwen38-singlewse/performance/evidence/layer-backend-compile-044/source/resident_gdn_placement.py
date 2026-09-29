"""Refine recurrent columns using the actual resident endpoint census.

Original matrix prefixes and all auxiliary identities remain immutable. New
cohosts absorb per-head deficits after existing slices shrink. Cost bounds for
changed widths/new roles are explicit estimates; a fresh whole-stage compile
and original-value reference proof remain mandatory.
"""
from collections import defaultdict
from copy import deepcopy

from spatial.gdn_placement import audit


def refine(original, banks, candidate, census, profiles, retained, *, guard=32,
           changed_width_reserve=64, new_port_reserve=2560, frontend_saving=96):
    audit(original,banks,candidate)
    total={tuple(c[:2]):c[2] for c in census['cells']}
    ends={tuple(p['pe']):p['bytes'] for p in candidate['bank_ends']}
    prefix={tuple(p['pe']):p['bytes'] for p in candidate['bank_prefixes']}
    prof={tuple(p['pe']):p for p in profiles}
    occupied={(*r['pe'],r['color'])for r in retained}
    if len(total)!=11388 or set(total)!=set(prof):
        raise ValueError('Complete actual endpoint census required')
    old_workers={tuple(w['pe']):w for w in candidate['workers']}
    changed_heads={w['head'] for pe,w in old_workers.items()
                   if total[pe]-ends[pe]+prefix[pe]+512*w['columns']>48128-guard}
    code={pe:total[pe]-ends[pe] for pe in ends}
    for pe in ends:
        if 'frontend_group' in prof[pe]['parameters']:
            code[pe]-=frontend_saving
        if pe in old_workers and old_workers[pe]['head'] in changed_heads:
            code[pe]+=changed_width_reserve
    limits={pe:48128-guard-code[pe] for pe in ends}
    heads=defaultdict(list);roles={};selected=[]
    for region in candidate['stage']['regions']:
        x,y,w,h=region['rect']
        for yy in range(y,y+h):
            for xx in range(x,x+w):roles[xx,yy]=region['role']
    for pe,old in old_workers.items():
        w=deepcopy(old)
        capacity=max(0,(limits[pe]-prefix[pe])//1024*2)
        w['columns']=min(w['columns'],capacity)
        if not w['columns']:
            raise ValueError('Existing state endpoint lost all capacity')
        heads[w['head']].append(w)
    new=[]
    for head in range(48):
        values=sorted(heads[head],key=lambda w:w['first'])
        deficit=128-sum(w['columns'] for w in values)
        if deficit<0 or deficit%2:
            raise ValueError('Invalid paired head partition')
        if deficit:
            choices=[]
            for pe,p in prof.items():
                if pe in old_workers or any(tuple(w['pe'])==pe for w in new) or pe not in ends:continue
                q=p['parameters'];name=p['source']
                ordinary=(name=='compact_layer_projection.csl' and
                          not q['root'] and not q.get('fusion_actor') and not q.get('mlp_sender'))
                mixer=(name=='device_mixer_layer_mlp_standby.csl' and not q['mixer_can_root'])
                if not (ordinary or mixer):continue
                colors={'gate_up':(18,20),'down':(11,12),'mix':(5,4)}[roles[pe]]
                if any((*pe,color)in occupied for color in colors):continue
                maximum=min(64,q['max_parts']*q['columns']//2) if ordinary else 64
                capacity=min(maximum,max(0,(limits[pe]-prefix[pe]-new_port_reserve)//1024*2))
                if capacity<deficit:continue
                distance=sum(abs(pe[0]-w['pe'][0])+abs(pe[1]-w['pe'][1]) for w in values)
                choices.append((distance,pe))
            if not choices:
                raise ValueError(f'No measured cohost can absorb head{head} deficit{deficit}')
            _,pe=min(choices)
            code[pe]+=new_port_reserve;limits[pe]-=new_port_reserve
            w=dict(pe=list(pe),columns=deficit,role=roles[pe],head=head,first=0,state_word=prefix[pe]//4)
            new.append(w);values.append(w)
        first=0
        for w in values:
            w['first']=first;first+=w['columns'];selected.append(w)
        if first!=128:
            raise ValueError('Incomplete original head')
    cursor=dict(prefix)
    for w in selected:
        pe=tuple(w['pe'])
        if 4*w['state_word']!=cursor[pe]:
            raise ValueError('Matrix prefix changed')
        cursor[pe]+=512*w['columns']
    if any(cursor[pe]>limits[pe] for pe in ends):
        raise ValueError('Protected prefix still exceeds the measured code bound')
    local=defaultdict(list);previous_owner={}
    for span in candidate['spans']:
        for index in range(span['pages']):
            key=span['namespace'],span['page_start']+index;pe=tuple(span['pe'])
            local[pe].append((span['byte_offset']+128*index,key));previous_owner[key]=pe
    capacity={pe:(limits[pe]-cursor[pe])//128 for pe in ends};owners={};spills=[]
    for pe,values in sorted(local.items()):
        for index,(_,key) in enumerate(sorted(values)):
            if index<capacity[pe]:owners[key]=pe
            else:spills.append(key)
        capacity[pe]-=min(len(values),capacity[pe])
    free={pe for pe,n in capacity.items() if n}
    for key in spills:
        if not free:raise ValueError('No space for retained auxiliary pages')
        x,y=previous_owner[key];pe=min(free,key=lambda p:(abs(x-p[0])+abs(y-p[1]),p))
        owners[key]=pe;capacity[pe]-=1
        if not capacity[pe]:free.remove(pe)
    spans=[]
    for (namespace,page),pe in sorted(owners.items()):
        offset=cursor[pe];cursor[pe]+=128
        if (spans and spans[-1]['namespace']==namespace and spans[-1]['pe']==list(pe) and
            spans[-1]['page_start']+spans[-1]['pages']==page and
            spans[-1]['byte_offset']+128*spans[-1]['pages']==offset):
            spans[-1]['pages']+=1
        else:spans.append(dict(namespace=namespace,page_start=page,pages=1,pe=list(pe),byte_offset=offset))
    result=deepcopy(candidate);result['workers']=selected;result['spans']=spans
    result['bank_ends']=[dict(pe=list(pe),bytes=n)for pe,n in sorted(cursor.items())]
    limits[tuple(original['request_controller']['pe'])]=0
    result['limits']=[dict(pe=list(pe),bytes=n)for pe,n in sorted(limits.items())]
    result['resident_refinement']=dict(source_attempt=census['attempt'],
        source_manifest_sha256=census['source_manifest_sha256'],census_sha256=census['census_sha256'],
        new_cohosts=new,changed_heads=sorted(changed_heads),auxiliary_pages_rehomed=len(spills),
        guard_bytes=guard,changed_width_code_reserve=changed_width_reserve,
        new_port_code_reserve=new_port_reserve,assumed_frontend_code_saving=frontend_saving,
        measured_frontend_saving_range=[112,128],
        original_matrix_prefixes_unchanged=True,full_compile_required=True,
        numerical_requalification_required=True,original_value_proof_required=True)
    result['audit']=audit(original,banks,result)
    return result
