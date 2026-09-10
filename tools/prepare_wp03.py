"""Freeze persistent-accumulation diagnostic304 or full5120, without SDK launch."""
import argparse,hashlib,json,shutil,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.stream_numerics import policy
from qwen38.stream_schedule import tiles


def prepare(root,run,image,slab,columns):
    if columns not in (304,5120):raise ValueError('Only explicitly qualified diagnostic/full profiles')
    run=Path(run).resolve();image=Path(image).resolve();slab=Path(slab)
    receipt=json.loads((slab/'download.json').read_text());raw=(slab/'slab-row-major.bf16').read_bytes()
    if not image.is_file() or receipt['status']!='complete' or len(raw)!=128*5120*2 or hashlib.sha256(raw).hexdigest()!=receipt['slab_sha256']:
        raise ValueError('Invalid existing image or slab')
    run.mkdir(parents=True,exist_ok=False);(run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir();(run/'tiles').mkdir()
    for name in ('pe.csl','layout.csl','driver.py','compile_entry.py'):shutil.copyfile(root/'examples/wp03'/name,run/name)
    shutil.copyfile(root/'csl/kernels/persistent_gemv_bf16_f32.csl',run/'kernel.csl')
    for p in (root/'core/qwen38').glob('*.py'):shutil.copyfile(p,run/'core/qwen38'/p.name)
    for name in ('guarded_run.py','sdk_container.py'):shutil.copyfile(root/'tools'/name,run/'tools'/name)
    shutil.copyfile(slab/'slab-row-major.bf16',run/'slab-row-major.bf16')
    (run/'slab-source.json').write_text(json.dumps(receipt,indent=2)+'\n')
    profile={'columns':columns,'rows':128,'tile_width':112,'calls':4,'onehot_index':columns-1,
             'kind':'full5120' if columns==5120 else 'diagnostic304_not_full_contraction',
             'kernel_timestamp':'48-bit cycle counter, three little-significance-first u16 words; elapsed modulo2^48; kernel-only scope',
             'tiles':[]}
    for tile in tiles(columns,1):
        chunks=[]
        for col in range(112):
            for row in range(128):
                offset=(row*5120+tile.offset+col)*2
                chunks.append(raw[offset:offset+2] if col<tile.valid else struct.pack('<H',0x3f80))
        packed=b''.join(chunks)
        restored=b''.join(packed[(col*128+row)*2:(col*128+row)*2+2] for row in range(128) for col in range(tile.valid))
        expected=b''.join(raw[(row*5120+tile.offset)*2:(row*5120+tile.offset+tile.valid)*2] for row in range(128))
        if restored!=expected:raise ValueError('Non-reversible tile pack')
        name=f'tiles/{tile.index:03d}.bf16';(run/name).write_bytes(packed)
        profile['tiles'].append({'index':tile.index,'offset':tile.offset,'valid':tile.valid,'file':name,
                                 'packed_sha256':hashlib.sha256(packed).hexdigest(),'valid_row_major_sha256':hashlib.sha256(restored).hexdigest()})
    (run/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (run/'numerical-policy.json').write_text(json.dumps(policy(columns),indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    steps={'planned_write_bytes':134217728,'steps':[
        {'name':'compile','seconds':300,'argv':prefix+['compile_entry.py','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']},
        {'name':'simulate','seconds':180 if columns==304 else 300,'argv':prefix+['driver.py']}]}
    (run/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps({'files':files,'scope':profile['kind']},indent=2)+'\n')
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--image',type=Path,required=True)
    p.add_argument('--slab',type=Path,required=True);p.add_argument('--columns',type=int,choices=(304,5120),required=True)
    a=p.parse_args();print(prepare(Path(__file__).resolve().parents[1],a.run,a.image,a.slab,a.columns))


if __name__=='__main__':main()
