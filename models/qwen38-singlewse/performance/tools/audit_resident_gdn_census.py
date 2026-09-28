"""Compare every measured resident PE with P45; preserve overflow diagnostics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def audit(attempt, output):
    if not re.fullmatch('layer-mlp-compile-[0-9]{3}', attempt):
        raise ValueError('Complete layer compiler attempt required')
    if output.exists():
        raise ValueError('Frozen census')
    folder = ROOT/'evidence'/attempt
    dispatch = json.loads((folder/'dispatch.json').read_text())
    remote = dispatch['remote']
    if remote != '/srv/model-storage/qwen38-singlewse/runs/'+attempt:
        raise ValueError('Unexpected invocation path')
    script = 'root='+repr(remote)+'\n'+'''from pathlib import Path
import json,hashlib
r=Path(root)
assert (r/'COMPLETE.json').exists() or (r/'FAILURE.json').exists()
raw=(r/'sram.json').read_bytes();assert len(raw)<16777216
s=json.loads(raw);cells=[]
assert s['application']==[78,146]
for record in s['records']:
 assert record['stack_allowance_bytes']==4096
 total=record['low_section_end']+4096
 assert record['passed']==(total<=48128)
 for x,y,w,h in record['rectangles']:
  for yy in range(y,y+h):
   for xx in range(x,x+w):cells.append([xx-s['offset'][0]+63,yy-s['offset'][1],total])
assert len(cells)==11388==len({(x,y) for x,y,_ in cells})
assert all(63<=x<141 and 0<=y<146 for x,y,_ in cells)
assert s['passed']==all(n<=48128 for x,y,n in cells)
print(json.dumps(dict(cells=cells,census_sha256=hashlib.sha256(raw).hexdigest(),
 source_manifest_sha256=hashlib.sha256((r/'source-manifest.json').read_bytes()).hexdigest(),
 elf_images=len(s['records']),passed=s['passed'])))
'''
    current = json.loads(subprocess.check_output(['ssh','workstation','python3 -c '+shlex.quote(script)],text=True,timeout=30))
    if current['source_manifest_sha256'] != hashlib.sha256((folder/'source-manifest.json').read_bytes()).hexdigest():
        raise ValueError('Measured source identity differs')
    manifest = json.loads((folder/'source-manifest.json').read_text())['files']
    def source(name):
        raw = (folder/'source'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest[name]:
            raise ValueError('Frozen source changed: '+name)
        return json.loads(raw)
    profiles = {tuple(pe):c for c in source('profiles.json')['profiles'] for pe in c['pes']}
    placement = source('gdn-bank-placement.json')
    ends = {tuple(p['pe']):p['bytes'] for p in placement['bank_ends']}
    protected = {tuple(p['pe']):p['bytes'] for p in placement['bank_prefixes']}
    for w in placement['workers']:
        pe=tuple(w['pe'])
        if protected[pe] != 4*w['state_word']:
            raise ValueError('State is not immediately after its immutable prefix')
        protected[pe] += 512*w['columns']
    base_path=ROOT/'evidence/resident-gdn-base-census-001.json'
    base=json.loads(base_path.read_text())
    base_folder=ROOT/'evidence/layer-mlp-compile-027'
    base_manifest_raw=(base_folder/'source-manifest.json').read_bytes()
    if hashlib.sha256(base_manifest_raw).hexdigest()!=base['source_manifest_sha256']:
        raise ValueError('Baseline census source identity differs')
    base_banks_raw=(base_folder/'source/gdn-bank-placement.json').read_bytes()
    if hashlib.sha256(base_banks_raw).hexdigest()!=json.loads(base_manifest_raw)['files']['gdn-bank-placement.json']:
        raise ValueError('Baseline bank identity differs')
    base_ends={tuple(p['pe']):p['bytes'] for p in json.loads(base_banks_raw)['bank_ends']}
    previous={tuple(c[:2]):c[2] for c in base['cells']}
    if set(previous)!=set(profiles):
        raise ValueError('Complete original coordinate identity')
    groups=defaultdict(list);bad=[]
    for x,y,total in current['cells']:
        pe=(x,y);c=profiles[pe]
        bank_delta=ends.get(pe,0)-base_ends.get(pe,0)
        groups[c['source']].append((total,total-previous[pe]-bank_delta,
                                    total-previous[pe],bank_delta))
        if total>48128:
            auxiliary=ends.get(pe,0)-protected.get(pe,0)
            bad.append(dict(pe=list(pe),source=c['source'],total=total,overflow=total-48128,
                            auxiliary_tail_bytes=auxiliary,
                            protected_prefix_deficit=max(0,total-auxiliary-48128),
                            parameters=c['parameters']))
    current.update(attempt=attempt,physical=False,executed=False,
                   original_census_sha256=base['census_sha256'],
                   original_census_receipt_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                   bad=bad,over_limit_pes=len(bad),
                   maximum_with_stack=max(c[2] for c in current['cells']),
                   delta_definition='Non-bank compiled footprint includes code, scratch and alignment; subtract actual per-PE bank-extent changes before comparing with027.',
                   code_delta_by_source={k:dict(pes=len(v),minimum=min(d for _,d,_,_ in v),
                       maximum=max(d for _,d,_,_ in v),maximum_with_stack=max(n for n,_,_,_ in v),
                       total_delta_min=min(d for _,_,d,_ in v),total_delta_max=max(d for _,_,d,_ in v),
                       bank_delta_min=min(d for _,_,_,d in v),bank_delta_max=max(d for _,_,_,d in v))for k,v in sorted(groups.items())},
                   scope='Complete local compiled-program census; no connected numerical or physical qualification.')
    output.write_text(json.dumps(current,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in current.items()if k not in ('cells','bad')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.attempt,args.output)
