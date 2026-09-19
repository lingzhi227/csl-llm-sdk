"""Actual finite topology, disjoint diagnostic/packet storage and SRAM qualification."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'core'))
from qwen38.elf_inventory import inventory,admit_wse3_sram
from cerebras.elf.cself import ELFMemory
from elf_symbols import symbols

def array(records,name,size):
    matches={}
    for r in records:
        n=r['name'].rsplit('$$',1)[-1] if r['name'].startswith('$$csl_base_address$$') else r['name']
        if (n==name or size==4 and n==name+'.0') and r['bytes']>0:matches[r['address'],r['bytes']]=r
    if len(matches)!=1:raise ValueError('One real array: '+name)
    v=next(iter(matches.values()))
    if v['bytes']!=size or v['address']%4:raise ValueError('Array extent/alignment: '+name)
    return v

def main():
    assert sorted(os.sched_getaffinity(0))==[0] and not (ROOT/'out').exists()
    args=['sdk_debug_shell','compile','layout.csl','--arch=wse3','--fabric-dims=185,4','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1']
    r=subprocess.run(args,timeout=300)
    if r.returncode:raise RuntimeError('Compile failed: '+str(r.returncode))
    covered=set();programs=[];files={}
    for p in sorted((ROOT/'out').rglob('*')):
        if p.is_file():
            raw=p.read_bytes();files[str(p.relative_to(ROOT))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    for p in sorted((ROOT/'out/bin').glob('*.elf')):
        raw=p.read_bytes();records=symbols(raw);reader=ELFMemory(p);assert tuple(reader.get_fabric_dimensions())==(185,4)
        rectangles=set(tuple(v) for v in reader.iter_rectangles());coords=set()
        for x,y,w,h in rectangles:coords|={(xx-4,yy-1) for xx in range(x,x+w) for yy in range(y,y+h)}
        assert coords and not coords&covered and all(0<=x<178 and 0<=y<2 for x,y in coords);covered|=coords
        capacities={192 if x==0 else 96 if x==2 else 3 if 40<=x<176 else 1 for x,y in coords};assert len(capacities)==1
        arrays=[array(records,'state',32),array(records,'diag.status',32),array(records,'diag.records',next(iter(capacities))*64)]
        if all(y==0 for x,y in coords):arrays += [array(records,'packets.header_buffer',4),array(records,'packets.receive_buffer',124)]
        if all(y==1 and (x in (0,2) or 40<=x<176) for x,y in coords):arrays.append(array(records,'diag.incoming',64))
        for i,a in enumerate(arrays):
            for b in arrays[i+1:]:assert max(a['address'],b['address'])>=min(a['address']+a['bytes'],b['address']+b['bytes']), (a,b)
        gate=admit_wse3_sram(inventory(raw),4096,48128)
        programs.append(dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),rectangles=sorted(rectangles),PEs=len(coords),arrays=arrays,sram=gate))
    assert covered=={(x,y) for x in range(178) for y in range(2)}
    result=dict(passed=all(v['sram']['passed'] for v in programs),files=files,application=programs,geometry=[185,4],application_shape=[178,2],PEs=356,runtime_created=False,DSR_allocation_proven=False)
    with (ROOT/'compiled.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    assert result['passed']
if __name__=='__main__':main()
