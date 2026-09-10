"""Fetch one 128x5120 original BF16 slab using the already verified WP01 header."""
import argparse,hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.ranged import RangeReader
from download_wp01_tile import MODEL,REVISION,SHARD,TENSOR,TOTAL_SIZE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wp01-tile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();old=json.loads((args.wp01_tile/'download.json').read_text())
    if (old['status'],old['model'],old['revision'],old['shard'],old['tensor'],old['tensor_shape']) != ('complete',MODEL,REVISION,SHARD,TENSOR,[17408,5120]):
        raise ValueError('Pinned WP01 source receipt mismatch')
    old_raw=(args.wp01_tile/'tile-row-major.bf16').read_bytes()
    if hashlib.sha256(old_raw).hexdigest()!=old['tile_row_major_sha256']:raise ValueError('Existing tile hash mismatch')
    base=old['tensor_absolute_start']
    if base!=8+old['header_bytes']+old['tensor_data_offsets'][0]:raise ValueError('Header offset mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    reader=RangeReader(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{SHARD}',TOTAL_SIZE,budget=4*1024*1024)
    report={'model':MODEL,'revision':REVISION,'shard':SHARD,'tensor':TENSOR,'dtype':'BF16',
            'tensor_shape':[17408,5120],'slab_shape':[128,5120],'tensor_absolute_start':base,
            'header_reused_sha256':old['header_sha256'],'wp01_receipt_sha256':hashlib.sha256((args.wp01_tile/'download.json').read_bytes()).hexdigest(),
            'whole_shard_hash_verified':False,'status':'started'}
    try:
        raw=reader.read(base,128*5120*2)
        prefix=b''.join(raw[r*5120*2:r*5120*2+112*2] for r in range(128))
        if prefix!=old_raw:raise ValueError('Full slab disagrees with original WP01 slice')
        (args.output/'slab-row-major.bf16').write_bytes(raw)
        report.update(status='complete',slab_sha256=hashlib.sha256(raw).hexdigest(),wp01_prefix_bit_equal=True)
    except BaseException as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        report.update(payload_bytes_received=reader.received,ranges=reader.receipts)
        (args.output/'download.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
