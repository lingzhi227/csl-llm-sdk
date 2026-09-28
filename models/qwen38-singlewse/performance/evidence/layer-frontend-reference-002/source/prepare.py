"""Remote bitwise migration and independent actual-descriptor readback.

Only exactly equal original FP32 MLP scales may alias. Preserve all mixer
matrices, local MLP auxiliary suffixes, dialogue pages and work slots.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from spatial.compact_mlp import CompactMlpPlacement, CompactMlpAuxiliaryPlacement

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()

def main():
    started=time.monotonic();read=lambda n:json.loads(Path(n).read_text())
    config=read('source-bank.json');root=Path(config['root'])
    for name,digest in config['hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Original fixture identity changed')
    original=read('frontend-bank-placement.json');plan=read('mlp-bank-placement.json')
    compact=CompactMlpPlacement(read('frontend-stage.json'))
    if compact.metadata()!=read('compact-mlp.json'):raise ValueError('Emitted compact MLP descriptors changed')
    right=CompactMlpAuxiliaryPlacement(original,compact,plan)
    oldindex=json.loads((root/'bank-index.json').read_text());old={tuple(r['pe']):r for r in oldindex['records']}
    source=np.load(root/'banks.npy',mmap_mode='r');source_ends={tuple(p['pe']):p['bytes'] for p in original['bank_ends'] if p['bytes']}
    if (source.dtype!=np.dtype('<u4') or source.ndim!=1 or source.nbytes!=original['allocated_bytes'] or
        len(old)!=len(oldindex['records']) or set(old)!=set(source_ends) or
        any(r['words']*4!=source_ends[pe] for pe,r in old.items())):raise ValueError('Source bank shape/identity')
    cursor=0
    for r in sorted(old.values(),key=lambda r:r['offset']):
        if r['offset']!=cursor:raise ValueError('Original bank overlap/gap')
        cursor+=r['words']
    if cursor!=source.size:raise ValueError('Original bank coverage')
    payload={tuple(pe):c['parameters'].get('bank_words',0)*4 for c in read('profiles.json')['profiles'] for pe in c['pes']}
    if payload!={tuple(p['pe']):p['bytes'] for p in plan['bank_ends']}:raise ValueError('Actual compiled bank extents changed')
    new={};cursor=0
    for pe,size in sorted(payload.items(),key=lambda kv:(kv[0][1],kv[0][0])):
        if size:new[pe]=dict(pe=list(pe),words=size//4,offset=cursor);cursor+=size//4
    if cursor*4!=plan['allocated_bytes'] or cursor*4>512<<20:raise ValueError('Candidate size bound')
    output=np.lib.format.open_memmap('banks.npy',mode='w+',dtype='<u4',shape=(cursor,))
    source_class=np.zeros(source.size,np.uint8);covered=np.zeros(cursor,np.bool_)
    source_scales=0;stored_scales=0;tiles=0

    def position(index,pe,offset,words):
        r=index[tuple(pe)]
        if offset<0 or words<1 or offset+words>r['words']:raise ValueError('Transfer exceeds bank')
        return r['offset']+offset

    def transfer(pe,first,last,words):
        i=position(old,pe,first,words);j=position(new,pe,last,words)
        if np.any(source_class[i:i+words]) or np.any(covered[j:j+words]):raise ValueError('Source/destination alias')
        output[j:j+words]=source[i:i+words];source_class[i:i+words]=1;covered[j:j+words]=True

    for pe,r in compact.records.items():
        iterations=r['iterations'];parts=r['parts'];blocks=r['scale_blocks']
        row_blocks=(r['first_output']%128+np.arange(iterations)*r['rows'])//128
        unique=np.array([np.flatnonzero(row_blocks==b)[0] for b in range(blocks)])
        for branch in range(r['branches']):
            i=position(old,pe,r['original_setup'][2+branch],iterations*parts*65)
            j=position(new,pe,r['setup'][2+branch],r['branch_words'])
            values=source[i:i+iterations*parts*65].reshape(iterations,parts,65)
            scales=values[:,:,64];shared=scales[unique,:]
            if not np.array_equal(scales,shared[row_blocks,:]):raise ValueError('Original scale aliases differ bitwise')
            if np.any(source_class[i:i+values.size]) or np.any(covered[j:j+r['branch_words']]):raise ValueError('Overlapping matrix branches')
            output[j:j+r['data_words']]=values[:,:,:64].reshape(-1)
            output[j+r['data_words']:j+r['branch_words']]=shared.reshape(-1)
            marks=source_class[i:i+values.size].reshape(iterations,parts,65);marks[:,:,:64]=1;marks[:,:,64]=2;marks[unique,:,64]=1
            covered[j:j+r['branch_words']]=True
            source_scales+=iterations*parts;stored_scales+=blocks*parts;tiles+=iterations*parts
    for p in original['bank_prefixes']:
        pe=tuple(p['pe']);r=compact.records.get(pe)
        old_start=r['original_bytes']//4 if r else 0;new_start=r['bytes']//4 if r else 0
        count=p['bytes']//4-old_start
        if count:transfer(pe,old_start,new_start,count)
    pages=0
    for span in original['spans']:
        for page in range(span['page_start'],span['page_start']+span['pages']):
            target=right.owner(page)
            if target['pe']!=span['pe']:raise ValueError('Auxiliary page changed PE')
            transfer(span['pe'],span['byte_offset']//4+32*(page-span['page_start']),target['byte_offset']//4,32);pages+=1
    aliases=source_scales-stored_scales
    if (not np.all(source_class) or not np.all(covered) or aliases*4!=plan['scale_alias_bytes'] or
        np.count_nonzero(source_class==1)!=cursor or np.count_nonzero(source_class==2)!=aliases):raise ValueError('Incomplete word partition')
    del covered,source_class;output.flush();del output
    actual=np.load('banks.npy',mmap_mode='r');verified_source_words=0
    before={tuple(pe):v for pe,v in read('original-worker-setups.json')};after={tuple(pe):v for pe,v in read('worker-setups.json')}
    profiles={tuple(pe):c['parameters'] for c in read('profiles.json')['profiles'] if c['source']=='compact_layer_projection.csl' for pe in c['pes']}
    if set(before)!=set(after) or set(after)!=set(profiles) or set(profiles)!=set(compact.records):raise ValueError('Actual setup coverage')
    # Independently use only actual setup fields and the CSL address formula.
    for pe,a in after.items():
        b=before[pe];p=profiles[pe];parts,iterations,base0,base1,first=a
        if a[:2]!=b[:2] or a[4]!=b[4] or not 0<parts<=p['max_parts']:raise ValueError('Logical MLP schedule changed')
        for branch in range(p['branches']):
            i=position(old,pe,b[2+branch],iterations*parts*65)
            values=source[i:i+iterations*parts*65].reshape(iterations,parts,65)
            base=a[2+branch];j=position(new,pe,base,iterations*parts*64)
            if not np.array_equal(values[:,:,:64],actual[j:j+iterations*parts*64].reshape(iterations,parts,64)):raise ValueError('Original weight codes changed')
            scale_offsets=base+iterations*parts*64+(((first%128+np.arange(iterations)*p['rows'])>>7)*parts)[:,None]+np.arange(parts)[None,:]
            if scale_offsets.min()<0 or scale_offsets.max()>=new[pe]['words']:raise ValueError('Actual CSL scale access exceeds bank')
            if not np.array_equal(values[:,:,64],actual[new[pe]['offset']+scale_offsets]):raise ValueError('Actual CSL reads wrong scale')
            verified_source_words+=values.size

    def verify(pe,first,last,words):
        nonlocal verified_source_words
        i=position(old,pe,first,words);j=position(new,pe,last,words)
        if not np.array_equal(source[i:i+words],actual[j:j+words]):raise ValueError('Original retained data changed')
        verified_source_words+=words
    for p in original['bank_prefixes']:
        pe=tuple(p['pe']);r=compact.records.get(pe);old_start=r['original_bytes']//4 if r else 0;new_start=r['bytes']//4 if r else 0
        if p['bytes']//4>old_start:verify(pe,old_start,new_start,p['bytes']//4-old_start)
    for a,b in zip(original['spans'],plan['spans']):verify(a['pe'],a['byte_offset']//4,b['byte_offset']//4,a['pages']*32)
    if verified_source_words!=source.size:raise ValueError('Fresh verification incomplete')
    index=dict(records=list(new.values()),total_words=cursor,allocated_bytes=cursor*4,
        initialized_weight_and_scale_bytes=oldindex['initialized_weight_and_scale_bytes']-aliases*4,
        remaining_bytes_are_zero_state_or_values=True,conversations=1,work_slots=2,
        bank_specialization='mlp-bank-placement.json',scale_encoding='Original mixer and MLP FP32 scales; verified local aliases only')
    Path('bank-index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
    result=dict(passed=True,physical=False,neural_execution=False,multi_turn_qualified=False,
        source_bytes=source.nbytes,allocated_bytes=cursor*4,verified_source_words=verified_source_words,
        original_mlp_tiles=tiles,retained_pages=pages,source_scale_words=source_scales,stored_scale_words=stored_scales,
        verified_scale_aliases=aliases,scale_alias_bytes=aliases*4,all_source_words_classified_once=True,
        all_original_values_verified_from_actual_csl_descriptors=True,no_requantization=True,
        hashes={n:sha(Path(n)) for n in ['banks.npy','bank-index.json']},seconds=time.monotonic()-started)
    Path('bank-remap-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
