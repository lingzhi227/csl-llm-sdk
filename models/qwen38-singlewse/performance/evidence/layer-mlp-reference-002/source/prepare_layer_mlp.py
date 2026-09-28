"""Freeze complete original layer0 banks and independent full-MLP reference.

Payloads stay on remote storage. Initial BF16 vectors are component fixtures,
not injected intermediate states of a claimed full-model inference run.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
from weights import OriginalWeights
from layer_schedule import tile_owner,auxiliary_owner,rank_xy
from bf16_silu_lut import build_and_prove
from mlp_oracle import mlp,bf16_bits


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    started=time.monotonic();stage=json.loads(Path('stage.json').read_text())
    profiles=json.loads(Path('profiles.json').read_text())['profiles'];metadata=json.loads(Path('tensors.json').read_text());hub=json.loads(Path('hub.json').read_text())
    assert stage['id']=='layer_00' and metadata['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
    pins={s['rfilename']:s['lfs']['sha256'] for s in hub['siblings'] if s['rfilename'].endswith('.safetensors')}
    reader=OriginalWeights('/srv/model-storage/qwen38-singlewse/model',metadata['tensors'],pins)
    payload={tuple(pe):p['parameters']['bank_words'] for p in profiles if 'bank_words' in p['parameters'] for pe in p['pes']}
    index=[];offsets={};cursor=0
    for pe,words in sorted(payload.items(),key=lambda x:(x[0][1],x[0][0])):
        if not words:continue
        offsets[pe]=cursor;index.append(dict(pe=list(pe),words=words,offset=cursor));cursor+=words
    if cursor*4>512<<20:raise ValueError('Bank fixture exceeds bounded file profile')
    bank=np.lib.format.open_memmap('banks.npy',mode='w+',dtype='<u4',shape=(cursor,));bank[:]=0
    occupied=np.zeros(cursor,np.bool_);matrix_records=[];original_bytes=0;resident_bytes=0
    def store(addresses,words):
        bases=np.asarray([offsets[tuple(a['pe'])]+a['byte_offset']//4 for a in addresses],dtype=np.int64)
        for a in addresses:
            if a['byte_offset']%4 or a['byte_offset']+a['bytes']>payload[tuple(a['pe'])]*4:raise ValueError('Original bank write outside compiled payload')
        positions=bases[:,None]+np.arange(words.shape[1],dtype=np.int64)[None,:]
        flat=positions.reshape(-1)
        if np.any(occupied[flat]) or len(np.unique(flat))!=len(flat):raise ValueError('Original bank writes overlap')
        bank[positions]=words;occupied[positions]=True
    for region in stage['regions']:
        for mi,matrix in enumerate(region['matrices']):
            m,k=matrix['shape'];rows,columns=matrix['tile_shape'];fp8=matrix['dtype']=='F8_E4M3';count_tiles=0;digest=hashlib.sha256();scale_digest=hashlib.sha256()
            for first in range(0,m,128):
                count=min(128,m-first);assert count%rows==0
                raw=reader.rows(matrix['tensor'],first,count);digest.update(raw.tobytes());original_bytes+=raw.nbytes
                tiles=np.ascontiguousarray(raw.reshape(count//rows,rows,k//columns,columns).transpose(0,2,3,1)).reshape(-1,rows*columns)
                if fp8:
                    if np.any((tiles&127)==127):raise ValueError('Original FP8 NaN')
                    scale=reader.rows(matrix['scale_tensor'],first//128,1)[0];scale_digest.update(scale.tobytes());original_bytes+=scale.nbytes
                    scale_bits=np.repeat(scale.astype(np.uint32)<<16,128//columns)
                    packed=np.empty((len(tiles),65),np.uint32);packed[:,:64]=tiles.copy().view('<u4')
                    packed[:,64]=np.tile(scale_bits,count//rows)
                else:packed=tiles.copy().view('<u4')
                tile_first=first//rows*(k//columns)
                addresses=[tile_owner(region,mi,tile_first+i) for i in range(len(tiles))]
                store(addresses,packed);count_tiles+=len(tiles);resident_bytes+=packed.nbytes
            assert count_tiles==matrix['tiles']
            matrix_records.append(dict(tensor=matrix['tensor'],shape=matrix['shape'],tiles=count_tiles,original_sha256=digest.hexdigest(),original_scale_sha256=scale_digest.hexdigest() if fp8 else None,tile_shape=matrix['tile_shape']))
            print(json.dumps(dict(phase='matrix_packed',**matrix_records[-1])),flush=True)
        for item in region['auxiliary']:
            if item['kind']!='weight':continue
            raw=reader.small_tensor(item['id']).astype('<u2',copy=False).tobytes();assert len(raw)==item['bytes']
            padded=np.zeros(item['pages']*128,np.uint8);padded[:len(raw)]=np.frombuffer(raw,np.uint8)
            addresses=[auxiliary_owner(region,item['page_start']+i) for i in range(item['pages'])]
            store(addresses,padded.view('<u4').reshape(-1,32));resident_bytes+=len(padded);original_bytes+=len(raw)
    assert int(occupied.sum())*4==resident_bytes;bank.flush();del occupied,bank
    Path('bank-index.json').write_text(json.dumps(dict(records=index,total_words=cursor,allocated_bytes=cursor*4,initialized_weight_and_scale_bytes=resident_bytes,remaining_bytes_are_zero_state_or_values=True),separators=(',',':'))+'\n')
    # Evict only consumed checkpoint cache pages before reference arithmetic.
    for shard in reader.verified:
        with (reader.root/shard).open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    table,proof=build_and_prove();np.save('silu-table.npy',table);Path('silu-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    rng=np.random.default_rng(383801)
    first=rng.normal(0,1,5120).astype(np.float32);first=np.float32(first/np.sqrt(np.mean(first.astype(np.float64)**2)))
    changed=rng.normal(0.05,0.35,5120).astype(np.float32)
    inputs=[bf16_bits(first),np.zeros(5120,np.uint16),bf16_bits(changed)];arrays={};cases=[]
    for case,operand in enumerate(inputs):
        def progress(value):print(json.dumps(dict(phase='reference',case=case,**value)),flush=True)
        result=mlp(reader,stage,operand,progress=progress)
        arrays[f'input_{case}']=operand;arrays[f'activation_{case}']=result['activation'];arrays[f'mixed_{case}']=result['mixed']
        for family in ['gate','up','down']:
            for field,value in result[family].items():arrays[f'{family}_{field}_{case}']=value
        cases.append(dict(index=case,epoch=case+1,mode=['normalized_nonzero','zero_after_nonzero','changed_nonzero'][case],replay_of=None))
    cases.append(dict(index=3,epoch=4,mode='warm_replay',replay_of=0))
    for name in list(arrays):
        if name.endswith('_0'):arrays[name[:-1]+'3']=arrays[name].copy()
    with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
    for shard,stamp in reader.verified.items():assert reader.stamp((reader.root/shard).stat())==stamp
    result=dict(passed=True,physical=False,device_execution=False,full_model=False,stage=stage['id'],model=metadata['repository'],revision=metadata['revision'],
        application=[78,146],controller=stage['request_controller']['pe'],original_mlp_shape=[5120,17408,5120],cases=cases,matrix_records=matrix_records,
        original_weight_bytes=original_bytes,resident_weight_and_expanded_scale_bytes=resident_bytes,allocated_bank_bytes=cursor*4,
        verified_original_shards={n:pins[n] for n in reader.verified},silu_proof=proof,
        hashes={n:sha(Path(n)) for n in ['banks.npy','bank-index.json','fixture.npz','silu-table.npy','silu-proof.json']},
        acceptance='Every original5120 BF16 output bit-exact to frozen ordered native FP32/full-K/PyTorch SiLU reference; all native input codes/scales and native/sender counters exact; all initialized original banks retained; four warm epochs including zero and replay; actual all-PE drain and normal stop.',
        projection_oracle='Independent row-major original weights, explicit32/64 FMA rounding and balanced central local-left-right FP32 reduction, every complete projection checked against FP64 mathematical truth and a priori gamma bounds.',
        scope='Complete original-weight MLP component; host supplies only initial BF16 boundary fixtures. Enclosing mixer weights are loaded but mixer/state operations are inactive. Not complete neural layer or model inference.',seconds=time.monotonic()-started)
    Path('fixture.json').write_text(json.dumps(result,indent=2)+'\n');Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,full_model=False,fixture_sha256=sha(Path('fixture.json')),seconds=result['seconds']))+'\n')
    print(json.dumps(dict(phase='complete',seconds=result['seconds'],allocated_bank_bytes=cursor*4)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
