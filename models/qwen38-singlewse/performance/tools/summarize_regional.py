"""Summarize complete paired regional evidence; cycles are never converted to TPS."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.evidence/'result.json').read_text());done=json.loads((a.evidence/'COMPLETE.json').read_text())
assert r['passed'] and r['normal_stop'] and not r['full_model']
if r['physical']:assert done['all_owned_jobs_released']
pairs={}
for row in r['cases']:
 pair=pairs.setdefault(row['case'],{});assert row['overlapped'] not in pair;pair[row['overlapped']]=row
rows=[]
for case,pair in sorted(pairs.items()):
 assert set(pair)=={False,True}
 s,o=[pair[b]['root_region_cycles'] for b in (False,True)]
 rows.append(dict(case=case,serialized_cycles=s,overlapped_cycles=o,cycles_saved=s-o,reduction_fraction=(s-o)/s,speedup=s/o))
sram=json.loads((a.evidence/'sram.json').read_text())
result=dict(physical=r['physical'],full_model=False,application=r['application'],same_pe_root_cycles=rows,
 max_compiled_sram_plus_declared_stack=max(x['low_section_end']+x['stack_allowance_bytes'] for x in sram['records']),
 scope=r['scope'],clock_hz=r['clock_hz'],timing_note=r['timing_note'],fixture_sha256=r['fixture_sha256'],
 interpretation='Measured whole-region schedule comparison, including launch-entry skew. The difference is not attributed solely to fabric transfer; callback scheduling also changes. No model rate or cross-PE timestamp subtraction.')
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
