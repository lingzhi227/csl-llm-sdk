"""Compare retained source fixtures across two tilings without changing either run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()


def reconstruct(target, rows, columns):
    assert target.shape==(960,128) and rows*columns==256
    result=np.empty((48,5120),np.uint16)
    for rank,tile in enumerate(target):
        group,k=divmod(rank,5120//columns)
        result[group*rows:(group+1)*rows,k*columns:(k+1)*columns]=tile.copy().view(np.uint16).reshape(columns,rows).T
    return result


def compare(reference,candidate):
    paths=[Path(reference),Path(candidate)];metadata=[];data=[]
    for p in paths:
        m=json.loads((p/'fixture.json').read_text())
        assert digest(p/'fixture.npz')==m['fixture_sha256']
        metadata.append(m)
        with np.load(p/'fixture.npz',allow_pickle=False) as f:
            data.append({k:f[k] for k in ['target','stream','bank']})
    old,new=metadata
    assert old['model']==new['model'] and old['revision']==new['revision']
    assert old['original_shards']==new['original_shards']
    assert old['full_matrix_shape']==new['full_matrix_shape']==[48,5120]
    oldmatrix=reconstruct(data[0]['target'],2,128)
    newmatrix=reconstruct(data[1]['target'],8,32)
    np.testing.assert_array_equal(oldmatrix,newmatrix)
    assert hashlib.sha256(newmatrix.tobytes()).hexdigest()==new['original_bf16_matrix_sha256']
    operands={}
    for c in [0,1,2,3,5]:
        assert old['cases'][c]['kind']==new['cases'][c]['kind']=='bf16'
        a=data[0]['stream'][c,:,:64].copy().reshape(-1)
        b=data[1]['stream'][c,:,:16].copy().reshape(-1)
        np.testing.assert_array_equal(a,b)
        h=hashlib.sha256(b.tobytes()).hexdigest();assert h==new['bf16_operand_sha256'][str(c)];operands[str(c)]=h
    # Preserve every other resident slot, including explicit zero padding.
    # Selected BF16 slot6 and FP8 slot0+scale are the only permitted replacements.
    unchanged_words=0
    for a,b in zip(old['workers'],new['workers']):
        for key in ['rank','xy','global_xy','bank_rank','eligible_rank','fp8_slots','bf16_slots','target_slot']:
            assert a[key]==b[key]
        rank=a['rank'];fp=a['fp8_slots'];begin=fp*65+6*128
        mask=np.ones(8814,dtype=bool);mask[:64]=False;mask[fp*64]=False;mask[begin:begin+128]=False
        np.testing.assert_array_equal(data[0]['bank'][rank,mask],data[1]['bank'][rank,mask]);unchanged_words+=int(mask.sum())
    return dict(schema='wse-retiled-fixture-comparison-v1',passed=True,physical=False,
                reference_fixture_sha256=old['fixture_sha256'],candidate_fixture_sha256=new['fixture_sha256'],
                original_bf16_matrix_sha256=new['original_bf16_matrix_sha256'],all_original_bf16_weights_identical=True,
                all_bf16_operands_identical=True,bf16_operand_sha256=operands,unchanged_background_and_padding_words=unchanged_words,
                coordinates_and_bank_capacities_identical=True,fp8_smoke_comparable=False,
                scope='Same complete original BF16 matrix and global inputs, same coordinates/capacities/background. Tile packing and ordered FP32 grouping differ. FP8 smoke weights differ and are not a performance pair.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=compare(a.reference,a.candidate)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
