"""Inspect actual connected fixture placement and one aligned weight bank."""
from pathlib import Path
import json
from cerebras.elf.cself import ELFMemory
from hw00.store import atomic_json
from elf_symbols import symbols

ROOT=Path(__file__).resolve().parent
width,height=json.loads((ROOT/'layout.json').read_bytes())['application']
if (width,height)!=(200,3):raise ValueError('Exact connected fixture geometry required')
plan=json.loads((ROOT/'transport-plan.json').read_bytes())
matrix_coordinates={(s['x']+ordinal+4,s['y']+1) for s in plan['stripes'] for ordinal in range(s['columns'])}
covered=set();rows=[]
for p in sorted((ROOT/'compiled/out/bin').glob('*.elf')):
    reader=ELFMemory(p)
    if tuple(reader.get_fabric_dimensions())!=(762,1172):raise ValueError('Hardware fabric mismatch')
    raw_count=0;unique=set()
    for r in reader.iter_rectangles():
        raw_count+=1
        if raw_count>width*height*32:raise ValueError('Raw ELF metadata bound')
        unique.add(tuple(r))
    rectangles=sorted(unique);coordinates=set()
    if not 0<len(rectangles)<=width*height:raise ValueError('Bounded nonempty actual placement')
    for x,y,w,h in rectangles:
        if w<1 or h<1 or x<4 or y<1 or x+w>width+4 or y+h>height+1:raise ValueError('Rectangle outside full MLP')
        for xx in range(x,x+w):
            for yy in range(y,y+h):
                coordinates.add((xx,yy))
    if coordinates & covered:raise ValueError('Overlapping placement between distinct ELFs')
    covered.update(coordinates)
    matrix=bool(coordinates & matrix_coordinates)
    if matrix and not coordinates<=matrix_coordinates:raise ValueError('Mixed matrix/nonmatrix program mapping')
    banks=[s for s in symbols(p.read_bytes()) if 'weight_storage' in s['name'] and s['bytes']==24576]
    if matrix:
        if len(banks)!=1 or banks[0]['address']%4:raise ValueError('One actual aligned24576-byte weight bank required')
    elif banks:raise ValueError('Unexpected matrix bank outside matrix placement')
    rows.append(dict(file=p.name,rectangles=rectangles,raw_rectangle_count=raw_count,weight_banks=banks))
if len(covered)!=width*height:raise ValueError('Full rectangle not completely covered')
atomic_json(ROOT/'placement.json',dict(passed=True,PEs=len(covered),matrix_PEs=len(matrix_coordinates),geometry=[762,1172],application=[width,height],files=rows,runtime_created=False,packed_alias_runtime_qualification_pending=True))
