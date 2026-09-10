"""Freeze one WP01 GEMV candidate using a tiny original tile or synthetic fixture."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.numerics import POLICY


def prepare(root,run,image,tile=None):
    run=Path(run).resolve(); image=Path(image).resolve()
    if not image.is_file(): raise ValueError('Existing SDK image required')
    if tile:
        tile=Path(tile);meta=json.loads((tile/'download.json').read_text())
        if meta['status']!='complete' or meta['reordering_reversible'] is not True:
            raise ValueError('Original tile download incomplete')
        raw=(tile/'tile-row-major.bf16').read_bytes();packed=(tile/'tile-column-major.bf16').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=meta['tile_row_major_sha256'] or hashlib.sha256(packed).hexdigest()!=meta['tile_column_major_sha256']:
            raise ValueError('Downloaded tile hash mismatch')
        source={'kind':'original_checkpoint_range','receipt':meta}
    else:
        # Exact BF16 dyadic values, generated locally; never called model weights.
        raw=b''.join(struct.pack('<I',struct.unpack('<I',struct.pack('<f',((r*3+c*5)%31-15)/32))[0])[2:]
                     for r in range(128) for c in range(112))
        packed=b''.join(raw[(r*112+c)*2:(r*112+c)*2+2] for c in range(112) for r in range(128))
        source={'kind':'synthetic_reproduction_fixture','formula':'((row*3+col*5)%31-15)/32'}
    if len(raw)!=28672 or len(packed)!=28672: raise ValueError('Wrong tile byte size')
    run.mkdir(parents=True,exist_ok=False)
    for name in ('layout.csl','pe.csl','driver.py','compile_entry.py'):
        shutil.copyfile(root/'examples/wp01'/name,run/name)
    shutil.copyfile(root/'csl/kernels/local_gemv_bf16_f32_colmajor.csl',run/'kernel.csl')
    (run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir()
    for p in (root/'core/qwen38').glob('*.py'):shutil.copyfile(p,run/'core/qwen38'/p.name)
    for name in ('guarded_run.py','sdk_container.py'):shutil.copyfile(root/'tools'/name,run/'tools'/name)
    (run/'tile-row-major.bf16').write_bytes(raw);(run/'tile-column-major.bf16').write_bytes(packed)
    (run/'tile-source.json').write_text(json.dumps(source,indent=2)+'\n')
    (run/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    steps={'planned_write_bytes':134217728,'steps':[
        {'name':'compile','seconds':300,'argv':prefix+['compile_entry.py','layout.csl','--arch=wse3','--fabric-dims=8,3',
          '--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']},
        {'name':'simulate','seconds':300,'argv':prefix+['driver.py']}]}
    (run/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps({'files':files,'scope':'WP01 128x112 BF16 GEMV, four calls'},indent=2)+'\n')
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--tile',type=Path)
    a=p.parse_args();print(prepare(Path(__file__).resolve().parents[1],a.run,a.image,a.tile))


if __name__=='__main__':main()
