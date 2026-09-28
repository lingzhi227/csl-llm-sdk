"""Freeze original gains and the complete residual/RMS/MLP/residual/RMS oracle.

Reuse immutable resident banks from the accepted MLP fixture. Read only original
checkpoint weights and synthetic subgraph boundary operands; no device captures
or candidate CSL arithmetic enter the expected arrays.
"""
import json
import os
import time
from pathlib import Path
import numpy as np
from weights import OriginalWeights
from mlp_oracle import mlp, bf16_bits
from norm_oracle import residual_rms
from prepare_layer_mlp import sha
from remap_banks import bank_segments
from layer_schedule import auxiliary_owner


def main():
    began=time.monotonic();binding=json.loads(Path('binding.json').read_text())
    source=Path(binding['bank_source']);base=json.loads((source/'fixture.json').read_text())
    if sha(source/'fixture.json')!=binding['base_fixture_sha256']:raise ValueError('Original bank fixture identity')
    stage=json.loads(Path('stage.json').read_text());metadata=json.loads(Path('tensors.json').read_text());hub=json.loads(Path('hub.json').read_text())
    if metadata['revision']!='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a':raise ValueError('Model revision')
    for name in ['banks.npy','bank-index.json','silu-table.npy','silu-proof.json']:
        if sha(source/name)!=base['hashes'][name]:raise ValueError('Immutable original payload')
        if name not in ['banks.npy','bank-index.json']:os.link(source/name,Path(name))
    remaps=json.loads(Path('state-page-remap.json').read_text())['pages']
    mixer=next(r for r in stage['regions'] if r['role']=='mix')
    items={i['id']:i for i in mixer['auxiliary']}
    for r in remaps:
        item=items[r['tensor']];expected=auxiliary_owner(mixer,r['page'])
        if item['kind']!='request_state' or item['id']!='recurrent.0':raise ValueError('Unexpected moved data')
        if not item['page_start']<=r['page']<item['page_start']+item['pages']:raise ValueError('State page identity')
        if r['source']!=expected['pe'] or r['source_byte_offset']!=expected['byte_offset']:raise ValueError('Original state source identity')
        if r['tensor_byte_offset']!=(r['page']-item['page_start'])*128:raise ValueError('Logical state coordinate')
    old_index=json.loads((source/'bank-index.json').read_text())
    plan=bank_segments(old_index['records'],json.loads(Path('profiles.json').read_text())['profiles'],remaps)
    old_bank=np.load(source/'banks.npy',mmap_mode='r',allow_pickle=False)
    new_bank=np.lib.format.open_memmap('banks.npy',mode='w+',dtype='<u4',shape=(plan['total_words'],))
    for part in plan['segments']:
        a=part['source_word'];b=part['destination_word'];length=part['words']
        if length>1<<18:raise ValueError('Bounded bank copy extent')
        new_bank[b:b+length]=old_bank[a:a+length]
    new_bank.flush();del old_bank,new_bank
    bank_index=dict(old_index,records=plan['records'],state_page_remap_count=len(remaps),all_original_words_preserved_once=True)
    Path('bank-index.json').write_text(json.dumps(bank_index,separators=(',',':'))+'\n')
    Path('bank-remap-proof.json').write_text(json.dumps(dict(passed=True,total_words=plan['total_words'],moved_pages=len(remaps),
        moved_bytes=sum(r['bytes'] for r in remaps),source_fixture_sha256=binding['base_fixture_sha256'],source_bank_sha256=base['hashes']['banks.npy'],
        output_bank_sha256=sha(Path('banks.npy')),source_and_destination_coverage='Each original word appears exactly once; only verified recurrent.0 state-page tails move.'),indent=2)+'\n')
    for name in ['banks.npy']:
        for path in [source/name,Path(name)]:
            with path.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    pins={s['rfilename']:s['lfs']['sha256'] for s in hub['siblings'] if s['rfilename'].endswith('.safetensors')}
    reader=OriginalWeights('/srv/model-storage/qwen38-singlewse/model',metadata['tensors'],pins)
    names=['model.language_model.layers.0.post_attention_layernorm.weight','model.language_model.layers.1.input_layernorm.weight']
    gains=np.stack([reader.small_tensor(name).astype(np.uint16,copy=False) for name in names])
    if gains.shape!=(2,5120):raise ValueError('Original norm gain shape')
    arrays={'norm_gains':gains};rng=np.random.default_rng(383831);cases=[];summaries=[]
    for case in range(3):
        left=bf16_bits(rng.normal(0,1 if case==0 else .35,5120).astype(np.float32))
        right=bf16_bits(rng.normal(0,.2,5120).astype(np.float32))
        if case==1:left[:]=0;right[:]=0
        before=residual_rms(left,right,gains[0])
        def progress(value):print(json.dumps(dict(phase='reference',case=case,**value)),flush=True)
        result=mlp(reader,stage,before['normalized'],progress=progress)
        after=residual_rms(before['residual'],result['down']['bf16'],gains[1])
        arrays.update({f'left_{case}':left,f'mixer_{case}':right,f'input_{case}':before['normalized'],
                       f'residual_{case}':after['residual'],f'successor_{case}':after['normalized'],
                       f'activation_{case}':result['activation'],f'mixed_{case}':result['mixed']})
        for label,norm in [('preceding',before),('successor',after)]:
            for field in ['fp32','fp64','bound','partials','inverse','bf16_fp64']:
                arrays[f'{label}_{field}_{case}']=np.asarray(norm[field])
        for family in ['gate','up','down']:
            for field,value in result[family].items():arrays[f'{family}_{field}_{case}']=value
        cases.append(dict(index=case,epoch=case+1,mode=['residual_nonzero','zero_after_nonzero','changed_nonzero'][case],replay_of=None))
        summaries.append(dict(case=case,preceding_maximum_error=float(np.max(np.abs(before['fp32'].astype(np.float64)-before['fp64']))),
                              successor_maximum_error=float(np.max(np.abs(after['fp32'].astype(np.float64)-after['fp64'])))))
    cases.append(dict(index=3,epoch=4,mode='warm_replay',replay_of=0))
    for name in list(arrays):
        if name.endswith('_0'):arrays[name[:-1]+'3']=arrays[name].copy()
    with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
    for shard,stamp in reader.verified.items():
        if reader.stamp((reader.root/shard).stat())!=stamp:raise ValueError('Original checkpoint changed')
        with (reader.root/shard).open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    record=dict(base,norm_bridge=True,cases=cases,norm_gain_tensors=names,norm_oracle_summaries=summaries,
        hashes={n:sha(Path(n)) for n in ['banks.npy','bank-index.json','fixture.npz','silu-table.npy','silu-proof.json']},
        verified_original_shards={n:pins[n] for n in reader.verified},
        acceptance='Original full MLP and successor RMS outputs bit-exact to independently frozen ordered FP32/BF16 oracle; both norms bounded against independent full-width FP64 truth before BF16; all native inputs/counters and forty residual owners exact; all banks, gains and LUTs retained; warm zero/change/replay, all-PE drain and normal stop.',
        scope='Actual residual/RMS -> complete original layer0 MLP -> residual/next-layer RMS subgraph. Host provides two initial BF16 boundary operands only. Original mixer/state banks retained but mixer/state neural execution and full model remain unqualified.',
        seconds=time.monotonic()-began)
    Path('fixture.json').write_text(json.dumps(record,indent=2)+'\n')
    Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,full_model=False,fixture_sha256=sha(Path('fixture.json')),seconds=record['seconds']))+'\n')
    print(json.dumps(dict(phase='complete',seconds=record['seconds'],scope=record['scope'])),flush=True)


if __name__=='__main__':main()
