"""Compact verified summary of a whole-region ELF census retained remotely."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(); parser.add_argument('attempt');parser.add_argument('--allow-failed',action='store_true'); args = parser.parse_args()
if not re.fullmatch('layer-((projection|mlp)-compile|mlp-hw)-[0-9]{3}', args.attempt): raise ValueError('Attempt')
physical='-hw-' in args.attempt
out = ROOT/'evidence'/args.attempt/'sram-summary.json'
if out.exists(): raise ValueError('Frozen summary')
remote = ('/srv/qwen38-singlewse-hardware/' if physical else '/srv/model-storage/qwen38-singlewse/runs/')+args.attempt
script = 'root='+repr(remote)+'\nallow_failed='+repr(args.allow_failed)+'\n'+'''import hashlib,json
from collections import Counter
from pathlib import Path
r=Path(root);raw=(r/'sram.json').read_bytes();assert len(raw)<16777216
s=json.loads(raw);actual=set();histogram=Counter();maximum=0;worst=[]
width,height=s['application'];ox,oy=s['offset']
for record in s['records']:
 assert record['stack_allowance_bytes']==4096
 total=record['low_section_end']+record['stack_allowance_bytes'];assert record['passed']==(total<=48128)
 if not allow_failed:assert record['passed']
 count=0
 for x,y,w,h in record['rectangles']:
  for yy in range(y,y+h):
   for xx in range(x,x+w):
    assert ox<=xx<ox+width and oy<=yy<oy+height and (xx,yy) not in actual
    actual.add((xx,yy));count+=1
 histogram[total]+=count
 if total>maximum:maximum=total;worst=[]
 if total==maximum and len(worst)<8:worst.append({k:record[k] for k in ['file','sha256','rectangles','low_section_end','stack_allowance_bytes']})
assert len(actual)==width*height==s['application_pes']
assert s['passed']==all(record['passed'] for record in s['records'])
print(json.dumps(dict(passed=s['passed'],over_limit_pes=sum(count for total,count in histogram.items() if total>48128),physical=False,executed=False,application=s['application'],application_pes=len(actual),
 compiled_elf_images=len(s['records']),maximum_with_stack=maximum,minimum_margin=48128-maximum,
 bytes_to_pe_count=dict(sorted(histogram.items())),maximum_examples=worst,
 remote_census_bytes=len(raw),remote_census_sha256=hashlib.sha256(raw).hexdigest(),
 source_manifest_sha256=hashlib.sha256((r/'source-manifest.json').read_bytes()).hexdigest(),
 note='Every ELF SRAM record and every original application coordinate independently checked. A false passed field rejects admission; full census and ELF images remain on remote storage.')))
'''
connection=['sh','/path/to/alcf-session.sh','host'] if physical else ['ssh','workstation']
result = subprocess.run(connection+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
data = json.loads(result.stdout)
with out.open('x') as f: json.dump(data,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in data.items() if k not in ['bytes_to_pe_count','maximum_examples']}))
