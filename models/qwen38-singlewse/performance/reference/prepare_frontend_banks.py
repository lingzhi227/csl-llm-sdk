"""Remote-only bitwise relocation of original mixer/MLP/dialogue banks.

Shared scale aliases must match all old original words. Fresh readback addresses
are independently reconstructed from the actual46-word CSL descriptors.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from spatial.compact_mixer import CompactMixerPlacement
from spatial.dialogue_placement import DialogueAuxiliaryPlacement
from spatial.frontend_placement import FrontendAuxiliaryPlacement
from spatial.layer_schedule import tile_owner


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    started=time.monotonic();read=lambda name:json.loads(Path(name).read_text())
    config=read('source-bank.json');root=Path(config['root'])
    for name,digest in config['hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Original fixture identity changed: '+name)
    oldindex=json.loads((root/'bank-index.json').read_text())
    old={tuple(r['pe']):r for r in oldindex['records']};source=np.load(root/'banks.npy',mmap_mode='r')
    original=next(r for r in read('original-stage.json')['regions'] if r['role']=='mix')
    previous=next(r for r in read('joint-stage.json')['regions'] if r['role']=='mix')
    dialogue=read('dialogue-bank-placement.json');placement=read('frontend-placement.json');plan=read('frontend-bank-placement.json')
    left=DialogueAuxiliaryPlacement(original,previous,read('joint-bank-placement.json'),dialogue)
    right=FrontendAuxiliaryPlacement(dialogue,placement,plan);compact=CompactMixerPlacement(placement['region'])
    if read('compact-mixer.json')!=compact.metadata():raise ValueError('Emitted compact descriptor metadata changed')
    source_ends={tuple(r['pe']):r['bytes'] for r in dialogue['bank_ends'] if r['bytes']}
    if (source.dtype!=np.dtype('<u4') or source.ndim!=1 or source.nbytes!=dialogue['allocated_bytes']
            or len(old)!=len(oldindex['records']) or set(old)!=set(source_ends)
            or any(r['words']*4!=source_ends[pe] for pe,r in old.items())):
        raise ValueError('Original bank index shape/extent mismatch')
    cursor=0
    for r in sorted(old.values(),key=lambda r:r['offset']):
        if r['offset']!=cursor:raise ValueError('Original bank gap or alias')
        cursor+=r['words']
    if cursor!=source.size:raise ValueError('Original bank index length')
    payload={tuple(pe):c['parameters'].get('bank_words',0)*4 for c in read('profiles.json')['profiles'] for pe in c['pes']}
    if payload!={tuple(r['pe']):r['bytes'] for r in plan['bank_ends']}:raise ValueError('Actual compiled bank extents differ')
    new={};cursor=0
    for pe,size in sorted(payload.items(),key=lambda kv:(kv[0][1],kv[0][0])):
        if size:new[pe]=dict(pe=list(pe),words=size//4,offset=cursor);cursor+=size//4
    if cursor*4!=plan['allocated_bytes'] or cursor*4>512<<20:raise ValueError('Candidate bank byte bound')
    output=np.lib.format.open_memmap('banks.npy',mode='w+',dtype='<u4',shape=(cursor,))
    source_class=np.zeros(source.size,np.uint8);covered=np.zeros(cursor,np.bool_)
    aliases=0;source_scales=0;stored_scales=0

    def position(index,address):
        r=index[tuple(address['pe'])];offset=address['byte_offset'];size=address['bytes']
        if offset<0 or offset%4 or size<=0 or size%4 or offset+size>4*r['words']:raise ValueError('Transfer outside declared bank')
        return r['offset']+offset//4,size//4

    def transfer(a,b,alias=False):
        nonlocal aliases,source_scales,stored_scales
        i,n=position(old,a);j,m=position(new,b)
        if n!=m or np.any(source_class[i:i+n]):raise ValueError('Repeated original word')
        if alias:
            if n!=1:raise ValueError('Only original FP32 scale words may alias')
            source_scales+=1
            if covered[j]:
                if output[j]!=source[i]:raise ValueError('Original scale words are not bitwise identical')
                aliases+=1;source_class[i]=2;return
            stored_scales+=1
        if np.any(covered[j:j+n]):raise ValueError('Destination data alias')
        output[j:j+n]=source[i:i+n];source_class[i:i+n]=1;covered[j:j+n]=True

    matrix_pes={tuple(r['pe']) for r in compact.records.values()}
    for p in dialogue['bank_prefixes']:
        if p['bytes'] and tuple(p['pe']) not in matrix_pes:
            address=dict(pe=p['pe'],byte_offset=0,bytes=p['bytes']);transfer(address,address)
    tiles=0
    for mi,m in enumerate(previous['matrices']):
        for tile in range(m['tiles']):
            a=tile_owner(previous,mi,tile);b=compact.tile(mi,tile)
            transfer(dict(pe=a['pe'],byte_offset=a['byte_offset'],bytes=256),dict(pe=b['pe'],byte_offset=b['byte_offset'],bytes=256))
            if m['dtype']=='F8_E4M3':
                transfer(dict(pe=a['pe'],byte_offset=a['byte_offset']+256,bytes=4),b['scale'],alias=True)
            tiles+=1
    pages=0
    for obj in plan['objects']:
        for page in range(obj['page_start'],obj['page_start']+obj['retained_pages']):
            transfer(left.owner(page),right.owner(page));pages+=1
    if (not np.all(source_class) or not np.all(covered) or aliases*4!=plan['scale_alias_bytes']
            or np.count_nonzero(source_class==1)!=cursor or np.count_nonzero(source_class==2)!=aliases):
        raise ValueError('Incomplete source/destination word partition')
    del source_class,covered
    output.flush();del output
    actual=np.load('banks.npy',mmap_mode='r');verified_source_words=0

    def verify(a,b):
        nonlocal verified_source_words
        i,n=position(old,a);j,m=position(new,b)
        if n!=m or not np.array_equal(source[i:i+n],actual[j:j+n]):raise ValueError('Original value changed on fresh readback')
        verified_source_words+=n

    for p in dialogue['bank_prefixes']:
        if p['bytes'] and tuple(p['pe']) not in matrix_pes:
            address=dict(pe=p['pe'],byte_offset=0,bytes=p['bytes']);verify(address,address)
    seen_tiles=[set() for _ in range(5)]
    setups=read('mixer-setups.json');setup_pes=set()
    for pe,setup in setups:
        if tuple(pe) in setup_pes or len(setup)!=46:raise ValueError('Descriptor owner/extent alias')
        setup_pes.add(tuple(pe))
        for mi,m in enumerate(previous['matrices']):
            base,count,first,stride,key,parts,bf16,rows=setup[mi*8:mi*8+8]
            if (parts,bf16,rows)!=(m['k_blocks'],int(m['dtype']=='BF16'),m['tile_shape'][0]):raise ValueError('Original arithmetic descriptor changed')
            for iteration in range(count):
                row=first+iteration*stride;tile=(row//rows)*parts+key
                if row%rows or tile in seen_tiles[mi] or not 0<=tile<m['tiles']:raise ValueError('Descriptor source coverage alias')
                seen_tiles[mi].add(tile);a=tile_owner(previous,mi,tile)
                verify(dict(pe=a['pe'],byte_offset=a['byte_offset'],bytes=256),dict(pe=pe,byte_offset=4*(base+iteration*64),bytes=256))
                if not bf16:
                    scale_word=setup[40+mi]+(row>>7)-(first>>7)
                    verify(dict(pe=a['pe'],byte_offset=a['byte_offset']+256,bytes=4),dict(pe=pe,byte_offset=4*scale_word,bytes=4))
    if any(len(seen_tiles[i])!=m['tiles'] for i,m in enumerate(previous['matrices'])):raise ValueError('Missing original descriptor tile')
    for obj in plan['objects']:
        for page in range(obj['page_start'],obj['page_start']+obj['retained_pages']):verify(left.owner(page),right.owner(page))
    if verified_source_words!=source.size:raise ValueError('Fresh source-word verification incomplete')
    index=dict(records=list(new.values()),total_words=cursor,allocated_bytes=cursor*4,
        initialized_weight_and_scale_bytes=oldindex['initialized_weight_and_scale_bytes']-aliases*4,
        remaining_bytes_are_zero_state_or_values=True,conversations=1,work_slots=2,
        bank_specialization='frontend-bank-placement.json',scale_encoding='Original FP32 block scales; only verified local aliases removed')
    Path('bank-index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
    result=dict(passed=True,physical=False,neural_execution=False,multi_turn_qualified=False,
        source_bytes=source.nbytes,allocated_bytes=cursor*4,verified_source_words=verified_source_words,
        original_mixer_tiles=tiles,retained_pages=pages,source_scale_words=source_scales,stored_scale_words=stored_scales,
        verified_scale_aliases=aliases,scale_alias_bytes=aliases*4,all_source_words_classified_once=True,
        all_original_values_verified_from_actual_csl_descriptors=True,no_requantization=True,
        hashes={n:sha(Path(n)) for n in ['banks.npy','bank-index.json']},seconds=time.monotonic()-started)
    Path('bank-remap-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
