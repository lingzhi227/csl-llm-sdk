"""Summarize complete spatial quantization and device consumer evidence in cycles."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.evidence/'result.json').read_text());done=json.loads((a.evidence/'COMPLETE.json').read_text())
assert r['passed'] and r['normal_stop'] and not r['full_model']
if r['physical']:assert done['all_owned_jobs_released'] and len(r['groups'])==44
sram=json.loads((a.evidence/'sram.json').read_text());assert sram['passed']
def stats(key):
 v=[g[key] for g in r['groups']];return dict(min=min(v),median=statistics.median(v),max=max(v),sum=sum(v))
result=dict(physical=r['physical'],full_model=False,groups=len(r['groups']),application=r['application'],items_per_pe=r['items_per_pe'],fixture_sha256=r['fixture_sha256'],scope=r['scope'],timing_note=r['timing_note'],clock_hz=r['clock_hz'],
 cycles={k:stats(k) for k in ['packet_present_cycles','producer_ready_cycles','native_consumer_cycles','producer_consumer_cycles']},
 host_seconds={k:stats(k) for k in ['host_arm_and_ready_seconds','host_start_through_result_audit_seconds']},
 weight_predecode_cycles=r['weight_predecode_cycles'],max_abs_fp64_error=max(g['max_abs_fp64_error'] for g in r['groups']),
 max_compiled_sram_plus_declared_stack=max(x['low_section_end']+x['stack_allowance_bytes'] for x in sram['records']),
 interpretation='Complete spatial group128 producer and one original2x128 consumer; no host work between them. Explicit host receive arming/readiness and one-time constant weight predecode are outside device interval and separately reported. Not full projection, model, or token rate.')
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
