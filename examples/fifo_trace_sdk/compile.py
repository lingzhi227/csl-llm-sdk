"""Compile the frozen small topology and inspect actual ELF placement and SRAM."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
from qwen38.elf_inventory import inventory,admit_wse3_sram
from cerebras.elf.cself import ELFMemory
from elf_symbols import symbols


def actual_array(raw, name, size):
    found = {}
    for item in symbols(raw):
        canonical = item['name'].rsplit('$$', 1)[-1] if item['name'].startswith('$$csl_base_address$$') else item['name']
        if (canonical == name or (size == 4 and canonical == name+'.0')) and item['bytes'] > 0:
            found[item['address'], item['bytes']] = item
    if len(found) != 1:
        raise ValueError('One real array required: '+name)
    item = next(iter(found.values()))
    if item['bytes'] != size or item['address'] % 4:
        raise ValueError('Exact aligned array extent: '+name)
    return item

def main():
    assert sorted(os.sched_getaffinity(0))==[0]
    info=json.loads((ROOT/'layout.json').read_bytes());width,height=info['application']
    assert (width,height)==(8,2) and not (ROOT/'out').exists()
    dims=(width+7,height+2)
    args=['sdk_debug_shell','compile','layout.csl','--arch=wse3',
          f'--fabric-dims={dims[0]},{dims[1]}','--fabric-offsets=4,1',
          '-o=out','--memcpy','--channels=1','--max-parallelism=1']
    completed=subprocess.run(args,timeout=60)
    if completed.returncode:raise RuntimeError('Compiler failed: '+str(completed.returncode))
    files={};application=[];covered=set()
    for p in sorted((ROOT/'out').rglob('*')):
        if p.is_file():
            raw=p.read_bytes();files[str(p.relative_to(ROOT))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    for p in sorted((ROOT/'out/bin').glob('*.elf')):
        reader=ELFMemory(p);rectangles=[tuple(v) for v in reader.iter_rectangles()]
        if tuple(reader.get_fabric_dimensions())!=dims:raise ValueError('Fabric dimensions')
        coords=set()
        for x,y,w,h in rectangles:
            coords|={(xx-4,yy-1) for xx in range(x,x+w) for yy in range(y,y+h)}
        if coords & covered:raise ValueError('Overlapping actual placement')
        if not coords or not all(0<=x<width and 0<=y<height for x,y in coords):raise ValueError('Outside application')
        covered|=coords
        raw=p.read_bytes();gate=admit_wse3_sram(inventory(raw),4096,48128)
        # Distinct x/y evidence constants make these placements role-identifiable.
        arrays={}
        for x,y in sorted(coords):
            capacity=64 if y==0 else 128
            expected={'evidence':640,'received_data':124,
                'packets.header_buffer':4,'packets.receive_buffer':124}
            expected['logger.storage' if x in (0,3,4) else 'empty_fifo']=capacity*4 if x in (0,3,4) else 512
            expected['sink.snapshot' if x in (2,5) else 'empty_trace']=640
            expected['side.snapshot' if (x,y)==(6,1) else 'empty_side']=128
            for name,size in expected.items():arrays[name]=actual_array(raw,name,size)
        values=list(arrays.values())
        for i,a in enumerate(values):
            for b in values[i+1:]:
                if max(a['address'],b['address']) < min(a['address']+a['bytes'],b['address']+b['bytes']):
                    raise ValueError('Actual evidence/FIFO/trace/packet arrays must be disjoint')
        application.append(dict(file=str(p.relative_to(ROOT)),rectangles=rectangles,sram=gate,arrays=arrays))
    if covered!={(x,y) for x in range(width) for y in range(height)}:raise ValueError('Incomplete physical placement')
    result=dict(passed=all(item['sram']['passed'] for item in application),files=files,
                application=application,geometry=dims,PEs=len(covered),runtime_created=False)
    with (ROOT/'compiled.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if not result['passed']:raise ValueError('Actual SRAM gate')

if __name__=='__main__':main()
