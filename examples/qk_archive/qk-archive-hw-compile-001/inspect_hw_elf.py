"""Inspect actual four-PE physical placements, original arrays and full-stack SRAM."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
from qwen38.elf_inventory import inventory, admit_wse3_sram
from cerebras.elf.cself import ELFMemory
from elf_symbols import symbols

SHARED_STORAGE = {
    'normalized_storage': (1024,2), 'trig_storage': (512,4),
    'rotary_stages_storage': (1536,4), 'product_casts_storage': (512,2),
    'trig_casts_storage': (128,2), 'probes_storage': (128,4),
}


def actual_array(records,name,size,alignment):
    found = {}
    for item in records:
        canonical = item['name'].rsplit('$$',1)[-1] if item['name'].startswith('$$csl_base_address$$') else item['name']
        if (canonical == name or size == 4 and canonical == name+'.0') and item['bytes'] > 0:
            found[item['address'],item['bytes']] = item
    if len(found) != 1:
        raise ValueError('One actual exported/leased array: '+name)
    item = next(iter(found.values()))
    if item['bytes'] != size or item['address'] % alignment:
        raise ValueError('Actual extent/alignment: '+name)
    return item


def main():
    if sorted(os.sched_getaffinity(0)) != [0] or (ROOT/'placement.json').exists():
        raise ValueError('CPU0 and one inspection attempt required')
    files = {}
    for path in sorted((ROOT/'compiled/out').rglob('*')):
        if path.is_file():
            if path.is_symlink() or path.stat().st_size > 8388608:
                raise ValueError('Bounded regular compiler output')
            raw = path.read_bytes()
            files[str(path.relative_to(ROOT))] = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    if len(files) > 64 or sum(v['bytes'] for v in files.values()) > 16777216:
        raise ValueError('Bounded four-PE compiler inventory')
    application = []
    covered = set()
    for path in sorted((ROOT/'compiled/out/bin').glob('*.elf')):
        raw = path.read_bytes()
        if len(raw)>3145728:
            raise ValueError('Bounded application ELF')
        reader = ELFMemory(path)
        if tuple(reader.get_fabric_dimensions()) != (762,1172):
            raise ValueError('Exact four-PE fabric')
        rectangles = set()
        for i,rectangle in enumerate(reader.iter_rectangles()):
            if i>=128:
                raise ValueError('Bounded raw placement metadata')
            rectangles.add(tuple(rectangle))
        coords = set()
        for x,y,w,h in rectangles:
            if not (4<=x<8 and y==1 and 1<=w<=4 and h==1 and x+w<=8):
                raise ValueError('Application rectangle')
            coords |= {(xx-4,0) for xx in range(x,x+w)}
        if not coords or coords & covered or len(coords) != 1:
            raise ValueError('Four distinct complete programs')
        covered |= coords
        role = next(iter(coords))[0]
        records = symbols(raw)
        expected = {'evidence':(256,4)}
        if role in (0,3):
            expected.update({name:(size,2) for name,size in {
                'attn.cache':8192,'attn.input':2048,'attn.qk_input':1024,
                'attn.qk_weights':1024,'attn.casts':1536,'output':512}.items()})
            expected['attn.qk_stats'] = 40,4
        if role==0:
            expected['attn.stages'] = 5120,4
        if role in (0,1):
            expected['observer.buffer'] = 128,4
        if role==1:
            expected.update({'observer.archive_storage':(3840,4),'observer.archive.status':(64,4),
                             'observer.snapshot':(128,4)})
        if role==3:
            expected['reference'] = 3840,4
            expected.update({'attn.'+name:spec for name,spec in SHARED_STORAGE.items()})
        if role==0:
            for item in records:
                if any(name in item['name'] for name in SHARED_STORAGE) and item['bytes']>4:
                    raise ValueError('Unexpected independent diagnostic bank in aliased owner')
        arrays = {name:actual_array(records,name,*spec) for name,spec in expected.items()}
        workspace = 'required actual 5120-byte aliased source workspace' if role==0 else 'not applicable'
        if role==3:
            workspace = 'optimized absent write-only workspace'
            if any(item['bytes']>0 and item['name'].rsplit('$$',1)[-1]=='attn.stages' for item in records):
                arrays['attn.stages'] = actual_array(records,'attn.stages',5120,4)
                workspace = 'retained 5120-byte write-only workspace'
        values = list(arrays.values())
        for i,left in enumerate(values):
            for right in values[i+1:]:
                if max(left['address'],right['address']) < min(left['address']+left['bytes'],right['address']+right['bytes']):
                    raise ValueError('Concurrent actual retained arrays overlap')
        gate = admit_wse3_sram(inventory(raw),4096,48128)
        application.append(dict(file=str(path.relative_to(ROOT)),role=role,
            rectangles=sorted(rectangles),sram=gate,arrays=arrays,workspace=workspace))
    if covered != {(x,0) for x in range(4)} or len(application)!=4:
        raise ValueError('Exactly four application programs')
    result = dict(passed=all(v['sram']['passed'] for v in application),files=files,
        application=application,PEs=4,geometry=[762,1172],runtime_created=False,
        scope='Actual allocations plus source-pinned alias offsets; actual alias/archive behavior still requires runtime')
    with (ROOT/'placement.json').open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    if not result['passed']:
        raise ValueError('Actual full-stack SRAM ceiling')


if __name__ == '__main__':
    main()
