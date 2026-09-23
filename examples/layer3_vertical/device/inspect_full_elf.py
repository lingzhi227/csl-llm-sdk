"""Measure actual full Layer3 placement and declared live banks, remotely."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
from elf_symbols import symbols
from fullgraph_contract import roles,expected_arrays
from qwen38.elf_inventory import inventory,admit_wse3_sram


def main():
    from cerebras.elf.cself import ELFMemory
    plan=json.loads((ROOT/'plan.json').read_bytes());transport=json.loads((ROOT/'transport-map.json').read_bytes())
    exports=json.loads((ROOT/'EXPORTS.json').read_bytes())['exports'];declared=roles(plan,transport)
    sram=json.loads((ROOT/'sram.json').read_bytes())['application_files']
    paths=sorted((ROOT/'compiled/out/bin').glob('*.elf'))
    assert 1<=len(paths)<=4096 and {p.name for p in paths}==set(sram)
    covered=set();programs=[];coordinate_banks=0
    for path in paths:
        assert path.stat().st_size<=4<<20
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();assert digest==sram[path.name]['sha256']
        reader=ELFMemory(path);assert tuple(reader.get_fabric_dimensions())==(762,1172)
        rects=set()
        for i,rect in enumerate(reader.iter_rectangles()):
            assert i<33640*32
            x,y,w,h=rect;assert 4<=x and 1<=y and w>=1 and h>=1 and x+w<=33 and y+h<=1161
            rects.add(tuple(rect))
        coords={(x-4+dx,y-1+dy) for x,y,w,h in rects for dx in range(w) for dy in range(h)}
        assert coords and coords<=set(declared) and not coords&covered;covered|=coords
        contracts={tuple(sorted(expected_arrays(declared[p],exports).items())) for p in coords}
        assert len(contracts)==1,'Actual program merges incompatible declared banks'
        records=symbols(raw);arrays={}
        for name,(size,alignment) in next(iter(contracts)):
            found={}
            for e in records:
                canonical=e['name'].rsplit('$$',1)[-1] if e['name'].startswith('$$csl_base_address$$') else e['name']
                if canonical in (name,name+'.0') and e['bytes']>0:found[e['address'],e['bytes']]=e
            assert len(found)==1,(path.name,name,'required actual array')
            e=next(iter(found.values()));assert e['bytes']==size and e['address']%alignment==0,(name,e,size)
            arrays[name]=e
        role_ids={declared[p]['role'] for p in coords};assert len(role_ids)==1
        if next(iter(role_ids)) in (4,8):assert len(coords)==1,'Each attention/head archive has its own fixed logical head identity'
        if next(iter(role_ids))==4:
            relocated=('normalized','trig','rotary_stages','product_casts','trig_casts','probes')
            for item in records:
                canonical=item['name'].rsplit('$$',1)[-1] if item['name'].startswith('$$csl_base_address$$') else item['name']
                assert not (canonical in {'attn.'+name for name in relocated} and item['type']==1 and item['bytes']>0),'Obsolete independent archive bank'
                assert not (any(('attn.'+name+'_storage') in canonical for name in relocated) and item['bytes']>4),'Unexpected fallback archive storage'
        values=list(arrays.values())
        for i,a in enumerate(values):
            for b in values[i+1:]:assert a['address']+a['bytes']<=b['address'] or b['address']+b['bytes']<=a['address'],(a['name'],b['name'])
        sections=inventory(raw);fit=admit_wse3_sram(sections,4096,48128)
        coordinate_banks+=len(coords)*len(arrays)
        programs.append(dict(file=path.name,bytes=len(raw),sha256=digest,rectangles=sorted(rects),PEs=len(coords),roles=sorted({declared[p]['role'] for p in coords}),arrays=arrays,sram=fit))
    assert covered==set(declared)
    report=dict(measurement_complete=True,application=[29,1160],fabric=[762,1172],offset=[4,1],PEs=33640,matrix_PEs=30576,router_PEs=639,coordinate_banks=coordinate_banks,programs=programs,all_measured_sram_passed=all(p['sram']['passed'] for p in programs),ceiling_with_4096_stack=48640,dynamic_stack_peak_proven=False,actual_task_DSR_lifetimes_qualified=False,host_copy_binding_qualified=False,neural_qualification=False,runtime_created=False,full_layer_accepted=False,full_model=False)
    raw=(json.dumps(report,separators=(',',':'))+'\n').encode();assert len(raw)<=32<<20
    with (ROOT/'compiled.json').open('xb') as f:f.write(raw)
    assert report['all_measured_sram_passed'],'Preserved actual measurement exceeds strict SRAM allowance'


if __name__=='__main__':main()
