"""Bounded SDK2.10 simulator ELF-core inspection; never model acceptance.

Only this4x3 probe in an11x5 simulator fabric is supported. Match each source
ELF's complete code section against its proposed core position before reading
any state. This bypasses the unavailable runtime symbol-table sidecar without
inventing addresses or changing the stopped core.
"""
import hashlib,json,struct
from pathlib import Path
from elftools.elf.elffile import ELFFile


def inspect(root):
 root=Path(root);core_path=root/'out.core'
 if core_path.stat().st_size>4<<20:raise ValueError('Unexpected probe core size')
 with core_path.open('rb') as stream:
  core=ELFFile(stream);segments=[(dict(s.header),s.data()) for s in core.iter_segments() if s['p_type']=='PT_LOAD']
 memory=next(data for h,data in segments if h['p_vaddr']==0 and h['p_memsz']==49152)
 if len(memory)!=11*5*49152:raise ValueError('Unexpected stopped fabric geometry')
 sram=json.loads((root/'sram.json').read_text())
 if (sram['application'],sram['fabric'],sram['offset'])!=([4,3],[11,5],[4,1]):raise ValueError('Unqualified core geometry')
 def read_config(pe,address,size):
  found=[(h,b) for h,b in segments if h['p_vaddr']<=address and address+size<=h['p_vaddr']+h['p_memsz']]
  if len(found)!=1:raise ValueError('Ambiguous or missing stopped register extent')
  h,b=found[0];first=(pe[1]*11+pe[0])*h['p_memsz']+address-h['p_vaddr']
  return b[first:first+size]
 observations=[];code_checks=[]
 requested=[((6,3),['device_submission','device_audit','device_submit_cursor','device_gateway_active','device_words','device_cursor','device_tx_done','device_rx_done'])]
 for worker in json.loads((root/'selection.json').read_text())['workers']:
  requested.append(((worker['pe'][0]+4,worker['pe'][1]+1),['sys.pending','sys.sending','sys.count','sys.cursor','sys.mode','sys.sequence','sys.have_half','sys.first_half']))
 for pe,names in requested:
  matches=[r for r in sram['records'] if any(x<=pe[0]<x+w and y<=pe[1]<y+h for x,y,w,h in r['rectangles'])]
  if len(matches)!=1:raise ValueError('Source ELF ownership differs from the frozen census')
  source=root/matches[0]['file'];raw=source.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=matches[0]['sha256']:raise ValueError('Source ELF changed')
  with source.open('rb') as stream:
   elf=ELFFile(stream);code=elf.get_section_by_name('.text');symbols=elf.get_section_by_name('.symtab')
   block=memory[(pe[1]*11+pe[0])*49152:(pe[1]*11+pe[0]+1)*49152]
   if block[code['sh_addr']:code['sh_addr']+code['sh_size']]!=code.data():raise ValueError('Stopped code does not match its proposed physical position')
   code_checks.append(dict(pe=pe,elf_sha256=matches[0]['sha256'],code_bytes=code['sh_size']))
   for name in names:
    items=symbols.get_symbol_by_name(name)
    if not items:
     # The SDK names a DSD-addressed global array through a base-address
     # symbol. Resolve only this exact compiler namespace and suffix, then
     # retain the resolved identity; never guess an SRAM address.
     items=[v for v in symbols.iter_symbols() if v.name.startswith('$$csl_base_address$$') and v.name.endswith('$$'+name)]
    if not items or len(items)!=1:raise ValueError('Missing or ambiguous symbol '+name)
    s=items[0];size=s['st_size'];address=s['st_value']
    if not 0<size<=24 or size%2 or address+size>49152:raise ValueError('Noncompact symbol extent')
    observations.append(dict(pe=pe,name=name,resolved_symbol=s.name,u16=list(struct.unpack('<'+'H'*(size//2),block[address:address+size]))))
 paths=[]
 for pe in [(4,3),(5,3),(6,3),(7,3)]:
  for color in [14,21,22,23]:
   value=struct.unpack('<H',read_config(pe,63488+color*2,2))[0]
   paths.append(dict(pe=pe,color=color,config=value,teardown=bool(value&32768)))
 return dict(diagnostic_only=True,physical=False,neural_execution=False,core_sha256=hashlib.sha256(core_path.read_bytes()).hexdigest(),
             core_bytes=core_path.stat().st_size,code_checks=code_checks,observations=observations,host_path_registers=paths)
