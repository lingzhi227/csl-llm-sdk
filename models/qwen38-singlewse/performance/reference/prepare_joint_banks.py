"""Conserve every original bank word and verify the actual new CSL descriptors.

Runs only on bounded remote storage. The independent source is the accepted
P31 original-weight bank fixture, with its recorded state-tail relocations.
Copy addressing uses tensor tile_owner; verification uses emitted CSL setup
descriptors. No neural computation or inference speed is claimed here.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from spatial.layer_schedule import tile_owner,auxiliary_owner
from spatial.joint_placement import JointAuxiliaryPlacement


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def main():
    started=time.monotonic();config=json.loads(Path('source-bank.json').read_text());oldroot=Path(config['root'])
    for name,digest in config['hashes'].items():
        if sha(oldroot/name)!=digest:raise ValueError('Original fixture identity changed: '+name)
    oldindex=json.loads((oldroot/'bank-index.json').read_text());old={tuple(r['pe']):r for r in oldindex['records']}
    source=np.load(oldroot/'banks.npy',mmap_mode='r');assert source.dtype==np.dtype('<u4')
    original=json.loads(Path('original-stage.json').read_text());candidate=json.loads(Path('joint-stage.json').read_text())
    before=next(r for r in original['regions'] if r['role']=='mix');after=next(r for r in candidate['regions'] if r['role']=='mix')
    plan=json.loads(Path('joint-bank-placement.json').read_text());resolver=JointAuxiliaryPlacement(before,after,plan)
    profiles=json.loads(Path('profiles.json').read_text())['profiles'];new={};cursor=0
    payload={tuple(pe):p['parameters'].get('bank_words',0) for p in profiles for pe in p['pes']}
    for pe,words in sorted(payload.items(),key=lambda p:(p[0][1],p[0][0])):
        if words:new[pe]=dict(pe=list(pe),words=words,offset=cursor);cursor+=words
    if cursor!=source.size or cursor*4!=plan['original_allocated_bytes'] or cursor*4>512<<20:raise ValueError('Original complete-bank byte extent changed')
    output=np.lib.format.open_memmap('banks.npy',mode='w+',dtype='<u4',shape=(cursor,))
    source_covered=np.zeros(cursor,np.bool_);destination_covered=np.zeros(cursor,np.bool_)
    copied=0;segments=0
    def position(index,address):
        record=index[tuple(address['pe'])];offset=address['byte_offset'];size=address['bytes']
        if offset%4 or size%4 or offset<0 or offset+size>record['words']*4:raise ValueError('Transfer outside original/candidate bank')
        return record['offset']+offset//4,size//4
    def transfer(left,right):
        nonlocal copied,segments
        a,n=position(old,left);b,m=position(new,right)
        if n!=m or np.any(source_covered[a:a+n]) or np.any(destination_covered[b:b+n]):raise ValueError('Original or candidate bank bytes overlap')
        output[b:b+n]=source[a:a+n];source_covered[a:a+n]=True;destination_covered[b:b+n]=True;copied+=n;segments+=1
    mx,my,mw,mh=before['rect']
    for pe,record in old.items():
        if mx<=pe[0]<mx+mw and my<=pe[1]<my+mh:continue
        transfer(dict(pe=list(pe),byte_offset=0,bytes=record['words']*4),dict(pe=list(pe),byte_offset=0,bytes=record['words']*4))
    for mi,matrix in enumerate(before['matrices']):
        for tile in range(matrix['tiles']):transfer(tile_owner(before,mi,tile),tile_owner(after,mi,tile))
        print(json.dumps(dict(phase='matrix_relocated',tensor=matrix['tensor'],tiles=matrix['tiles'])),flush=True)
    moved={m['page']:dict(pe=m['destination'],byte_offset=m['destination_byte_offset'],bytes=128)
           for m in json.loads(Path('old-state-page-remap.json').read_text())['pages']}
    for page in range(before['auxiliary_pages']):transfer(moved.get(page) or auxiliary_owner(before,page),resolver.owner(page))
    if copied!=cursor or not np.all(source_covered) or not np.all(destination_covered):raise ValueError('Missing original or candidate resident words')
    del source_covered,destination_covered
    output.flush();del output
    actual=np.load('banks.npy',mmap_mode='r');setups=json.loads(Path('mixer-setups.json').read_text());verified=0;digests=[]
    for mi,matrix in enumerate(before['matrices']):
        digest=hashlib.sha256();tile_bytes=256 if matrix['dtype']=='BF16' else 260
        # The arithmetic program consumes base+iteration*tile_words and reports
        # first_row+iteration*row_stride. Read exactly those emitted descriptors,
        # not the candidate tile_owner that generated the transfers above.
        for pe,setup in setups:
            base,count,first,stride,key,parts,bf16,rows=setup[mi*8:mi*8+8]
            if (parts,bf16,rows)!=(matrix['k_blocks'],int(matrix['dtype']=='BF16'),matrix['tile_shape'][0]):raise ValueError('Actual descriptor arithmetic extent')
            for iteration in range(count):
                row=first+iteration*stride;original_tile=(row//rows)*parts+key
                a,n=position(old,tile_owner(before,mi,original_tile))
                b,m=position(new,dict(pe=pe,byte_offset=4*base+iteration*tile_bytes,bytes=tile_bytes))
                if n!=m or not np.array_equal(source[a:a+n],actual[b:b+n]):raise ValueError('Actual CSL descriptor selects different original weights/scales')
                digest.update(actual[b:b+n].tobytes());verified+=1
        digests.append(dict(tensor=matrix['tensor'],descriptor_order_sha256=digest.hexdigest()))
    if verified!=sum(m['tiles'] for m in before['matrices']):raise ValueError('Incomplete descriptor verification')
    for page in range(before['auxiliary_pages']):
        a,n=position(old,moved.get(page) or auxiliary_owner(before,page));b,m=position(new,resolver.owner(page))
        if n!=m or not np.array_equal(source[a:a+n],actual[b:b+n]):raise ValueError('Original auxiliary page changed')
    for pe,record in old.items():
        if mx<=pe[0]<mx+mw and my<=pe[1]<my+mh:continue
        a=record['offset'];b=new[pe]['offset'];n=record['words']
        if not np.array_equal(source[a:a+n],actual[b:b+n]):raise ValueError('Retained original MLP bank prefix changed')
    index=dict(records=list(new.values()),total_words=cursor,allocated_bytes=cursor*4,
        initialized_weight_and_scale_bytes=oldindex['initialized_weight_and_scale_bytes'],remaining_bytes_are_zero_state_or_values=True)
    Path('bank-index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
    result=dict(passed=True,physical=False,neural_execution=False,all_source_and_destination_words_covered_once=True,
        allocated_bytes=cursor*4,transfers=segments,original_mixer_tiles_verified=verified,original_auxiliary_pages_verified=before['auxiliary_pages'],
        original_mlp_prefixes_verified=True,descriptor_digests=digests,source_hashes=config['hashes'],
        hashes={name:sha(Path(name)) for name in ['banks.npy','bank-index.json']},seconds=time.monotonic()-started)
    Path('bank-remap-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
