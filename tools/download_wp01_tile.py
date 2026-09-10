"""Fetch exactly 128 row slices (28,672 bytes) plus bounded safetensors header."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.ranged import RangeReader

MODEL='Qwen/Qwen3.8-27B'
REVISION='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
SHARD='model-00001-of-00018.safetensors'
TENSOR='model.language_model.layers.0.mlp.up_proj.weight'
TOTAL_SIZE=3966730552
ROWS,COLS=128,112


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output
    out.mkdir(parents=True,exist_ok=False)
    reader=RangeReader(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{SHARD}',TOTAL_SIZE)
    report={'model':MODEL,'revision':REVISION,'shard':SHARD,'tensor':TENSOR,
            'whole_shard_hash_verified':False,'status':'started'}
    def save():
        report.update(payload_bytes_received=reader.received, ranges=reader.receipts)
        (out/'download.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        header_len=struct.unpack('<Q',reader.read(0,8))[0]
        if header_len > 1024*1024: raise ValueError('Header budget exceeded')
        header=reader.read(8,header_len)
        tensor=json.loads(header)[TENSOR]
        if tensor['dtype']!='BF16' or tensor['shape']!=[17408,5120]:
            raise ValueError('Pinned tensor shape/dtype mismatch')
        begin,end=tensor['data_offsets']
        if end-begin != 17408*5120*2: raise ValueError('Tensor extent mismatch')
        base=8+header_len+begin
        rows=[]
        # Serial, bounded requests: minimum payload, no speculative prefetch.
        for row in range(ROWS):
            rows.append(reader.read(base+row*5120*2,COLS*2))
            if row % 16 == 0:
                print('Downloaded rows',row+1,'of',ROWS,flush=True); save()
        raw=b''.join(rows)
        packed=b''.join(raw[(r*COLS+c)*2:(r*COLS+c)*2+2] for c in range(COLS) for r in range(ROWS))
        restored=b''.join(packed[(c*ROWS+r)*2:(c*ROWS+r)*2+2] for r in range(ROWS) for c in range(COLS))
        if restored != raw: raise ValueError('Tile transpose not reversible')
        (out/'tile-row-major.bf16').write_bytes(raw)
        (out/'tile-column-major.bf16').write_bytes(packed)
        report.update(status='complete',dtype='BF16',tensor_shape=tensor['shape'],tensor_data_offsets=[begin,end],
            header_bytes=header_len,header_sha256=hashlib.sha256(header).hexdigest(),
            tensor_absolute_start=base,tile_rows=[0,ROWS],tile_columns=[0,COLS],
            tile_row_major_sha256=hashlib.sha256(raw).hexdigest(),
            tile_column_major_sha256=hashlib.sha256(packed).hexdigest(),reordering_reversible=True)
        save()
    except BaseException as exc:
        report.update(status='failed',error=str(exc)); save(); raise


if __name__=='__main__': main()
