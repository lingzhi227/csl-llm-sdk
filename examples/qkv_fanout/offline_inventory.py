"""Read only actual ELF symbols and captured core inventory; never run a device."""
import hashlib,json,os
from pathlib import Path
from cerebras.elf.cself import ELFLoader,ELFMemory
ROOT=Path.cwd()
assert sorted(os.sched_getaffinity(0))==[0]
result=json.loads((ROOT/'result.json').read_bytes())
assert result['normal_stop']
retained={'files':result['core_files']}
receipt=json.loads((ROOT/'compiled.json').read_bytes())
files=[]
for name,v in retained['files'].items():
 p=ROOT/name;assert p.is_file() and not p.is_symlink() and p.stat().st_size==v['bytes']
 files.append(dict(name=name,**v))
programs=[]
for spec in receipt['application']:
 p=ROOT/spec['file'];raw=p.read_bytes();bound=receipt['files'][spec['file']]
 assert len(raw)==bound['bytes'] and hashlib.sha256(raw).hexdigest()==bound['sha256']
 loader=ELFLoader(elf_file=str(p));mapping={}
 for name,value in loader.symbols.items():
  if any(word in name for word in ['state','metadata','data','counts','sending','receiving','active','waiting','pending','fanout','finished','entered','done','offset','destination','length']):
   address,size=value
   if 0<=address<49152 and 0<size<=4096:mapping[name]=dict(byte_address=address,bytes=size)
 programs.append(dict(file=spec['file'],rectangles=sorted(set(tuple(v) for v in spec['rectangles'])),symbols=mapping))
raw=(json.dumps(dict(core_files=files,programs=programs),indent=2)+'\n').encode()
assert len(raw)<=262144
with (ROOT/'offline-inventory.json').open('xb') as f:f.write(raw)
print(json.dumps(dict(status='inventory_saved',programs=len(programs),bytes=len(raw))))
