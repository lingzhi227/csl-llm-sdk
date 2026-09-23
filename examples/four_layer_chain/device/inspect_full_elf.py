"""Bounded streaming inspection of the actual original0-3 compile archive."""
from pathlib import Path
import hashlib,json,struct,sys,tarfile,tempfile
from source_gate import ROOT,PROFILE
from fullgraph_contract import roles,expected_arrays
from elf_symbols import symbols
from hw00.store import atomic_json
sys.path.insert(0,str(ROOT/'core'))
from qwen38.elf_inventory import inventory,admit_wse3_sram


def sections(raw):
 if raw[4]==2:off=struct.unpack_from('<Q',raw,40)[0];stride,count,_=struct.unpack_from('<HHH',raw,58);fmt='<IIQQQQIIQQ'
 else:off=struct.unpack_from('<I',raw,32)[0];stride,count,_=struct.unpack_from('<HHH',raw,46);fmt='<10I'
 assert raw[5]==1 and count<=256 and off+stride*count<=len(raw)
 return [struct.unpack_from(fmt,raw,off+i*stride) for i in range(count)]


def main():
 from cerebras.elf.cself import ELFMemory
 artifact=json.loads((ROOT/'artifact.json').read_bytes());path=Path(artifact['artifact'])
 assert path.parent.resolve()==ROOT.resolve() and not path.is_symlink() and path.stat().st_size==artifact['bytes']<=PROFILE['host_file_bytes']
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==artifact['sha256']
 source=json.loads((ROOT/'manifest.json').read_bytes())['files'];source_csl={n:v for n,v in source.items() if n.endswith('.csl')}
 declared=roles();declared_points=set(declared);covered=set();programs=[];bank_count=0;seen=set();embedded=set();total_bytes=0;member_count=0
 # Exactly one temporary ELF is materialized at a time. The original immutable
 # archive is the retained executable evidence; no complete extraction copy.
 with tempfile.TemporaryDirectory(prefix='inspection-',dir=ROOT) as scratch:
  scratch=Path(scratch)
  with tarfile.open(path,'r|gz') as archive:
   for member in archive:
    member_count+=1;total_bytes+=member.size
    assert member_count<=PROFILE['archive_members'] and total_bytes<=PROFILE['archive_uncompressed_bytes']
    assert member.name not in seen and not member.issym() and not member.islnk();seen.add(member.name)
    if not member.isfile():continue
    matches=[n for n in source_csl if member.name.endswith('/csl/'+n)]
    if matches:
     assert len(matches)==1;n=matches[0];assert n not in embedded and member.size==source_csl[n]['bytes']
     raw=archive.extractfile(member).read(member.size+1);assert len(raw)==member.size and hashlib.sha256(raw).hexdigest()==source_csl[n]['sha256'];embedded.add(n)
    if '/out/bin/' not in member.name or not member.name.endswith('.elf'):continue
    assert len(programs)<PROFILE['maximum_ELFs'] and 0<member.size<=4<<20
    raw=archive.extractfile(member).read(member.size+1);assert len(raw)==member.size
    digest=hashlib.sha256(raw).hexdigest();temporary=scratch/'current.elf';temporary.write_bytes(raw)
    reader=ELFMemory(temporary);assert tuple(reader.get_fabric_dimensions())==(762,1172)
    rects=set()
    for i,rect in enumerate(reader.iter_rectangles()):
     assert i<119*1160*32
     x,y,w,h=rect;assert 4<=x and 1<=y and w>=1 and h>=1 and x+w<=123 and y+h<=1161
     rects.add(tuple(rect))
    del reader
    coords={(x-4+dx,y-1+dy) for x,y,w,h in rects for dx in range(w) for dy in range(h)}
    assert coords and coords<=declared_points and not coords&covered;covered|=coords
    identities={(declared[p]['family'],declared[p]['layer_id'],declared[p]['role']) for p in coords};assert len(identities)==1
    family,layer_id,role=next(iter(identities))
    contracts={tuple(sorted(expected_arrays(declared[p]).items())) for p in coords};assert len(contracts)==1
    records=symbols(raw);arrays={}
    for name,(size,alignment) in next(iter(contracts)):
     found={}
     for e in records:
      canonical=e['name'].rsplit('$$',1)[-1] if e['name'].startswith('$$csl_base_address$$') else e['name']
      if canonical in (name,name+'.0') and e['bytes']>0:found[e['address'],e['bytes']]=e
     assert len(found)==1,(member.name,name,'Actual retained bank required')
     e=next(iter(found.values()));assert e['bytes']==size and e['address']%alignment==0,(member.name,name,e,size)
     arrays[name]=e
    values=list(arrays.values())
    for i,a in enumerate(values):
     for b in values[i+1:]:assert a['address']+a['bytes']<=b['address'] or b['address']+b['bytes']<=a['address'],(member.name,a['name'],b['name'])
    if family=='attention' and role in (4,8):assert len(coords)==1
    # Prove each actual program was specialized for its own original layer,
    # including routers/idle PEs, by reading the initialized semantic control.
    bank=arrays['checkpoint_control'];sec=sections(raw)[bank['section']]
    assert sec[1]==1 and sec[3]<=bank['address'] and bank['address']+32<=sec[3]+sec[5]
    offset=sec[4]+bank['address']-sec[3];control=list(struct.unpack_from('<8I',raw,offset))
    sample=declared[next(iter(coords))]
    assert control==[0x51435031,sample['request_id'],sample['stage_id'],layer_id,1,0,1,0],(member.name,control)
    footprint=inventory(raw);physical=admit_wse3_sram(footprint,4096,49152);legacy=admit_wse3_sram(footprint,4096,PROFILE['legacy_family_ceilings'][family])
    bank_count+=len(coords)*len(arrays)
    programs.append(dict(file=Path(member.name).name,archive_member=member.name,bytes=len(raw),sha256=digest,rectangles=sorted(rects),PEs=len(coords),family=family,layer_id=layer_id,role=role,arrays=arrays,initial_checkpoint_control=control,sections=footprint,sram=physical,legacy_family_sram=legacy))
    temporary.unlink()
 assert covered==declared_points and embedded==set(source_csl),(len(covered),set(source_csl)-embedded)
 passed=all(p['sram']['passed'] for p in programs);legacy_failures=[dict(file=p['file'],layer=p['layer_id'],role=p['role'],bytes_with_stack=p['sram']['low_section_end']+4096,old_ceiling=p['legacy_family_sram']['ordinary_address_ceiling']) for p in programs if not p['legacy_family_sram']['passed']]
 report=dict(measurement_complete=True,application=[119,1160],fabric=[762,1172],offset=[4,1],PEs=len(covered),matrix_PEs=125220,coordinate_banks=bank_count,programs=programs,all_measured_sram_passed=passed,physical_ceiling_with_4096_stack=49152,minimum_physical_margin=min(49152-p['sram']['low_section_end']-4096 for p in programs),legacy_family_failures=legacy_failures,archive_member_count=member_count,archive_uncompressed_bytes=total_bytes,embedded_source_files=len(embedded),dynamic_stack_peak_proven=False,actual_task_DSR_lifetimes_qualified=False,host_copy_binding_qualified=False,neural_qualification=False,runtime_created=False,full_model=False)
 raw=(json.dumps(report,separators=(',',':'))+'\n').encode();assert len(raw)<=32<<20
 with (ROOT/'compiled.json').open('xb') as f:f.write(raw)
 atomic_json(ROOT/'sram.json',dict(passed=passed,legacy_family_passed=not legacy_failures,legacy_family_failures=legacy_failures,physical_ceiling=49152,stack_allowance=4096,minimum_physical_margin=report['minimum_physical_margin'],programs=len(programs),PEs=len(covered),artifact=artifact))
 assert passed,'Actual physical capacity including unchanged4096-byte stack exceeded'
 print(json.dumps({k:v for k,v in report.items() if k not in ('programs','legacy_family_failures')}))

if __name__=='__main__':main()
