from pathlib import Path
import json,hashlib
r=Path('/srv/model-storage/qwen38-singlewse/runs/layer-mlp-compile-017')
s=json.loads((r/'sram.json').read_text());profiles=json.loads((r/'profiles.json').read_text())['profiles'];ox,oy=s['offset'];stage=json.loads((r/'joint-stage.json').read_text());sx,sy,*_=stage['rect']
by_pe={}
for v in s['records']:
 for x,y,w,h in v['rectangles']:
  for xx in range(x,x+w):
   for yy in range(y,y+h):
    assert (xx-ox+sx,yy-oy+sy) not in by_pe
    by_pe[xx-ox+sx,yy-oy+sy]=v
out={}
for p in profiles:
 key=p['source'];g=out.setdefault(key,dict(pes=0,maximum_with_stack=0,minimum_with_stack=100000,minimum_margin=48128))
 for pe in p['pes']:
  v=by_pe[tuple(pe)];n=v['low_section_end']+v['stack_allowance_bytes'];assert v['passed'];g['pes']+=1;g['maximum_with_stack']=max(g['maximum_with_stack'],n);g['minimum_with_stack']=min(g['minimum_with_stack'],n);g['minimum_margin']=min(g['minimum_margin'],48128-n)
assert sum(v['pes'] for v in out.values())==len(by_pe)==11388
print(json.dumps(dict(roles=out,source_manifest_sha256=hashlib.sha256((r/'source-manifest.json').read_bytes()).hexdigest(),remote_census_sha256=hashlib.sha256((r/'sram.json').read_bytes()).hexdigest(),physical=False,executed=False)))
