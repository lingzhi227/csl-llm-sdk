"""Compile the frozen small topology and inspect actual ELF placement and SRAM."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
from elf_inventory import inventory,admit_wse3_sram
from cerebras.elf.cself import ELFMemory

def main():
    assert sorted(os.sched_getaffinity(0))==[0]
    info=json.loads((ROOT/'layout.json').read_bytes());width,height=info['application']
    assert (width,height)==(37,10) and not (ROOT/'out').exists()
    dims=(width+7,height+2)
    args=['sdk_debug_shell','compile','layout.csl','--arch=wse3',
          f'--fabric-dims={dims[0]},{dims[1]}','--fabric-offsets=4,1',
          '-o=out','--memcpy','--channels=1','--max-parallelism=1']
    completed=subprocess.run(args,timeout=120)
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
        gate=admit_wse3_sram(inventory(p.read_bytes()),4096,48128)
        application.append(dict(file=str(p.relative_to(ROOT)),rectangles=rectangles,sram=gate))
    if covered!={(x,y) for x in range(width) for y in range(height)}:raise ValueError('Incomplete physical placement')
    result=dict(passed=all(item['sram']['passed'] for item in application),files=files,
                application=application,geometry=dims,PEs=len(covered),runtime_created=False)
    with (ROOT/'compiled.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if not result['passed']:raise ValueError('Actual SRAM gate')

if __name__=='__main__':main()
