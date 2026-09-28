"""Read-only qualification of new native packing against original layer0/1 files.

All selected shard hashes/headers are checked, and original row/scale boundaries
are sampled for every matrix. This does not upload or execute a neural layer.
"""
import hashlib,json,struct,sys,time
from pathlib import Path
import numpy as np
sys.path[:0]=['performance','runtime','performance/runtime']
from weights import OriginalWeights
from layer_weights import matrix_tiles
from spatial.layer_schedule import tile_owner,auxiliary_owner
from source_gate import verify


def main():
    verify();start=time.monotonic();root=Path('.')
    tensors=json.loads((root/'configs/tensors.json').read_text())['tensors']
    hub=json.loads((root/'configs/hub.json').read_text())
    pins={s['rfilename']:s['lfs']['sha256'] for s in hub['siblings'] if s['rfilename'].endswith('.safetensors')}
    weights=OriginalWeights('/srv/model-storage/qwen38-singlewse/model',tensors,pins)
    plan=json.loads((root/'selected-layer0-1.json').read_text());records=[];small=[];samples=bytes_total=0
    if [s['layer'] for s in plan['stages']]!=[0,1]:raise ValueError('Actual adjacent original layers required')
    for stage in plan['stages']:
        for region in stage['regions']:
            for index,matrix in enumerate(region['matrices']):
                rows,columns=matrix['tile_shape'];m,k=matrix['shape'];fp8=matrix['dtype']=='F8_E4M3'
                blocks=sorted({b for b in [0,127//rows,128//rows,m//rows-1] if 0<=b<m//rows})
                raw_row=None;row_index=None;scale_row=None;scale_index=None;digest=hashlib.sha256();count=0
                for tile,payload in matrix_tiles(weights,matrix,blocks):
                    block,column=divmod(tile,k//columns);first=block*rows;column*=columns
                    if row_index!=block:raw_row=weights.rows(matrix['tensor'],first,rows);row_index=block
                    original=raw_row[:,column:column+columns]
                    if fp8:
                        if scale_index!=first//128:
                            scale_index=first//128;scale_row=weights.rows(matrix['scale_tensor'],scale_index,1)
                        # Independent bulk transpose vs the packer's scalar column/row loop.
                        expected=original.T.copy(order='C').tobytes()+struct.pack('<I',int(scale_row[0,column//128])<<16)
                    else:expected=original.astype('<u2',copy=False).tobytes()
                    if payload!=expected:raise ValueError('Original native weight/scale bytes differ')
                    address=tile_owner(region,index,tile)
                    if address['pe']==stage['request_controller']['pe'] or address['byte_offset']+len(payload)>plan['config']['payload_per_pe']:
                        raise ValueError('Initialization writes outside its resident owner')
                    digest.update(payload);count+=1;samples+=1;bytes_total+=len(payload)
                records.append(dict(tensor=matrix['tensor'],tile_shape=[rows,columns],sampled_row_blocks=blocks,
                                    sampled_tiles=count,packed_sample_sha256=digest.hexdigest(),passed=True))
            for item in region['auxiliary']:
                if item['kind']!='weight':continue
                raw=weights.small_tensor(item['id']).astype('<u2',copy=False).tobytes()
                if len(raw)!=item['bytes']:raise ValueError('Original small-weight extent')
                for page in range(item['pages']):
                    address=auxiliary_owner(region,item['page_start']+page)
                    if address['pe']==stage['request_controller']['pe']:raise ValueError('Controller aliases original auxiliary weight')
                small.append(dict(tensor=item['id'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    verify()
    for shard,stamp in weights.verified.items():
        if weights.stamp((weights.root/shard).stat())!=stamp:raise ValueError('Original checkpoint changed during audit')
    result=dict(passed=True,physical=False,neural_execution=False,device_uploaded=False,layers=[0,1],
                matrix_records=records,small_weights=small,sampled_tiles=samples,sampled_native_bytes=bytes_total,
                verified_original_shards={name:pins[name] for name in weights.verified},seconds=time.monotonic()-start,
                scope='Original shard identity plus selected row/block/scale boundaries for every layer0/1 matrix and complete small parameters. Not an exhaustive packed-bank readback.')
    with Path('COMPLETE.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,matrices=len(records),small_parameters=len(small),sampled_tiles=samples,seconds=result['seconds'])))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
