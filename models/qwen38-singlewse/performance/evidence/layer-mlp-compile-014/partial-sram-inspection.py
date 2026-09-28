import resource,signal,time,json,sys,fcntl,hashlib
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456));resource.setrlimit(resource.RLIMIT_CPU,(30,30));signal.alarm(35)
f=open('/srv/cerebras-workstation/heavy.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
r=Path('/srv/model-storage/qwen38-singlewse/runs/layer-mlp-compile-014');sys.path.insert(0,str(r))
from elf_inventory import inventory,admit_wse3_sram
from placement import placement
profiles=json.loads((r/'profiles.json').read_text())['profiles'];by_pe={tuple(pe):p for p in profiles for pe in p['pes']};seen=set();classes={};fail=[];t=time.monotonic()
for path in r.glob('out/bin/*.elf'):
 raw=path.read_bytes();gate=admit_wse3_sram(inventory(raw),stack_allowance=4096,ceiling=48128);total=gate['low_section_end']+4096
 for x,y,w,h in placement(raw,(85,148))['rectangles']:
  for yy in range(y,y+h):
   for xx in range(x,x+w):
    pe=(xx-4+63,yy-1);assert pe in by_pe and pe not in seen;seen.add(pe);p=by_pe[pe]
    key=(p['source'],p['parameters'].get('bank_words',0));item=classes.setdefault(key,dict(source=key[0],bank_words=key[1],min_bytes=total,max_bytes=total,pes=0,over_limit=0))
    item['min_bytes']=min(item['min_bytes'],total);item['max_bytes']=max(item['max_bytes'],total);item['pes']+=1;item['over_limit']+=int(total>48128)
    if total>48128:fail.append(dict(pe=pe,bytes=total,bank_words=key[1],source=key[0]))
missing=[dict(pe=pe,source=by_pe[pe]['source'],parameters=by_pe[pe]['parameters']) for pe in sorted(set(by_pe)-seen)]
print(json.dumps(dict(complete=False,accepted=False,physical=False,compile='layer-mlp-compile-014',source_manifest_sha256=hashlib.sha256((r/'source-manifest.json').read_bytes()).hexdigest(),linked_pes=len(seen),missing_pes=missing,over_limit_pes=fail,classes=list(classes.values()),seconds=time.monotonic()-t)))
