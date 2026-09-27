"""Admission/precision evidence for the mixed resident bank, no token-rate estimate."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.evidence/'result.json').read_text());done=json.loads((a.evidence/'COMPLETE.json').read_text());sram=json.loads((a.evidence/'sram.json').read_text())
assert r['passed'] and r['normal_stop'] and r['all_weights_retained'] and r['weight_uploads']==1 and not r['full_model'] and len(r['cases'])==12
if r['physical']:assert done['all_owned_jobs_released']
assert sram['passed'] and sram['application_pes']==6
groups={}
for c in r['cases']:
 kind='mixed' if len(set(c['types']))>1 else 'bf16' if c['types'][0] else 'fp8'
 groups.setdefault(kind,[]).append(c['root_contraction_cycles'])
maximum=max(x['low_section_end']+x['stack_allowance_bytes'] for x in sram['records'])
out=dict(physical=r['physical'],full_model=False,full_input_width=False,full_output_rows=False,application=r['application'],fp8_slots_per_pe=112,bf16_slots_per_pe=12,control_bytes_per_pe=4,per_tile_metadata_bytes=0,
 max_compiled_sram_plus_declared_stack=maximum,sram_margin_bytes=48128-maximum,cycle_ranges={k:[min(v),max(v)] for k,v in groups.items()},epochs=len(r['cases']),weight_uploads=1,
 max_abs_fp64_error=max(c['max_abs_fp64_error'] for c in r['cases']),fixture_sha256=r['fixture_sha256'],scope=r['scope'],timing_note=r['timing_note'],clock_hz=None,
 interpretation='Actual mixed-path SRAM and resident lifecycle qualification, not full-model ownership/placement admission. Root timing includes representative sixPE tree, with different arithmetic by kind; no precision speedup claim from comparing them.')
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
