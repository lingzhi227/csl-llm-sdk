candidate='layer-mlp-compile-024'

import hashlib,json,resource
from collections import Counter,defaultdict
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20));resource.setrlimit(resource.RLIMIT_CPU,(45,45))
root=Path('/srv/model-storage/qwen38-singlewse/runs')
before=root/'layer-mlp-compile-022';after=root/candidate
def read(p,n):return json.loads((p/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def census(p):
 s=read(p,'sram.json');assert s['passed'] and s['application']==[78,146]
 ox,oy=s['offset'];found={}
 for r in s['records']:
  assert r['passed'] and r['stack_allowance_bytes']==4096
  total=r['low_section_end']+4096;assert total<=48128
  for x,y,w,h in r['rectangles']:
   for yy in range(y,y+h):
    for xx in range(x,x+w):
     pe=(xx-ox+63,yy-oy);assert 63<=pe[0]<141 and 0<=pe[1]<146 and pe not in found
     found[pe]=(r['source'],total)
 assert len(found)==11388==s['application_pes'];return found
def profiles(p):
 found={}
 for c in read(p,'profiles.json')['profiles']:
  for pe in c['pes']:
   key=tuple(pe);assert key not in found;found[key]=(c['source'],c['parameters'])
 assert len(found)==11388;return found
for p in [before,after]:
 assert read(p,'COMPLETE.json')['sram_passed']
 for n,d in read(p,'source-manifest.json')['files'].items():assert sha(p/n)==d
neural=[p.name for p in before.glob('*.csl') if p.name!='layout.csl']
assert all((before/n).read_bytes()==(after/n).read_bytes() for n in neural)
unchanged=['frontend-stage.json','frontend-bank-placement.json','frontend-placement.json','mixer-setups.json','device-network.json','network-binding.json']
assert all((before/n).read_bytes()==(after/n).read_bytes() for n in unchanged)
assert (before/'layout.csl').read_bytes().split(b' const route_0=',1)[1].rstrip()==(after/'layout.csl').read_bytes().split(b' const route_0=',1)[1].rstrip()
old,new=census(before),census(after);op,np=profiles(before),profiles(after)
assert old.keys()==new.keys()==op.keys()==np.keys()
roles=defaultdict(list);gains=Counter();cells=[]
for pe in sorted(old):
 role,prior=old[pe];assert role==op[pe][0] and new[pe][0]==np[pe][0]
 assert new[pe][0]==('compact_layer_projection.csl' if role=='layer_projection.csl' else role)
 a,b=op[pe][1],np[pe][1]
 assert {k:v for k,v in a.items() if k!='bank_words'}=={k:v for k,v in b.items() if k!='bank_words'}
 bank_gain=4*(a.get('bank_words',0)-b.get('bank_words',0));margin=48128-new[pe][1]
 roles[new[pe][0]].append(margin);gains[prior-new[pe][1]]+=1
 cells.append([*pe,new[pe][0],margin,prior-new[pe][1],bank_gain])
assert sum(c[5] for c in cells)==3912128
print(json.dumps(dict(passed=True,physical=False,neural_execution=False,complete_model_speed=False,
 source_attempt=before.name,candidate_attempt=after.name,application_pes=len(cells),
 unchanged_original_neural_csl_files=len(neural),all_existing_routes_unchanged=True,
 bank_bytes_reclaimed=sum(c[5] for c in cells),actual_sram_bytes_reclaimed=sum(c[4] for c in cells),minimum_margin=min(c[3] for c in cells),
 sram_delta_to_pe_count=dict(sorted(gains.items())),
 roles={r:dict(pes=len(v),minimum_margin=min(v),maximum_margin=max(v)) for r,v in roles.items()},
 cell_fields=['x','y','role','actual_margin_bytes','actual_sram_gain_bytes','bank_gain_bytes'],cells=cells,
 identities={p.name:{n:sha(p/n) for n in ['source-manifest.json','sram.json']} for p in [before,after]},
 scope='Complete actual resident-layer compilation with4096B stack. Original MLP scale sharing only; new state workers, their transport and full neural/dialogue speed remain unqualified.'),separators=(',',':')))
