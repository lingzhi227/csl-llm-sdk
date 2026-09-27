"""Full-K component comparison; each schedule has its own frozen numerical oracle."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.evidence/'result.json').read_text());done=json.loads((a.evidence/'COMPLETE.json').read_text());assert r['passed'] and r['normal_stop'] and not r['full_model']
if r['physical']:assert done['all_owned_jobs_released'] and len(r['cases'])==8
pairs={}
for c in r['cases']:
 for row in c['roots']:
  pair=pairs.setdefault((c['case'],row['k_blocks']),{});assert c['schedule'] not in pair;pair[c['schedule']]=row['root_contraction_cycles']
comparison=[]
for (case,k),pair in sorted(pairs.items()):
 assert set(pair)=={'tree','chain'};s,t=pair['chain'],pair['tree'];comparison.append(dict(case=case,k_blocks=k,chain_cycles=s,tree_cycles=t,reduction_fraction=(s-t)/s,component_speedup=s/t))
sram=json.loads((a.evidence/'sram.json').read_text());assert sram['passed']
out=dict(physical=r['physical'],full_model=False,full_output_rows=False,application=r['application'],active_matrix_pes=sum(r['blocks']),full_k_blocks=r['blocks'],fixture_sha256=r['fixture_sha256'],comparisons=comparison,
 max_compiled_sram_plus_declared_stack=max(x['low_section_end']+x['stack_allowance_bytes'] for x in sram['records']),
 max_abs_fp64_error=max(v['max_abs_fp64_error'] for c in r['cases'] for v in c['roots']),scope=r['scope'],timing_note=r['timing_note'],clock_hz=None,
 interpretation='Full input widths, only two output rows per projection. Both schedules satisfy separate exactFP32 and FP64 bounds; regrouping changes addition order. No dynamic quantization, full-M/model throughput, assumed clock, or cross-PE subtraction.')
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
