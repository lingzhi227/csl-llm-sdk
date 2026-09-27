"""Autonomous control-loop evidence with same-controller timing, never model TPS."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.evidence/'result.json').read_text());done=json.loads((a.evidence/'COMPLETE.json').read_text());sram=json.loads((a.evidence/'sram.json').read_text());fixture=json.loads((a.evidence/'fixture.json').read_text())
assert r['passed'] and r['normal_stop'] and r['all_banks_oracles_requests_retained'] and r['weight_uploads']==1 and not r['full_model'] and len(r['runs'])==2 and sum(x['rounds'] for x in r['runs'])==192
if r['physical']:assert done['all_owned_jobs_released']
assert sram['passed'] and sram['application_pes']==8
groups={};gaps=[]
for run in r['runs']:
 assert run['autonomous_total_cycles']==sum(run['request_to_result_cycles'])+sum(run['inter_epoch_cycles'])
 for case,cycles in zip(run['sequence'],run['request_to_result_cycles']):groups.setdefault('bf16' if fixture['cases'][case]['kind'] else 'fp8',[]).append(cycles)
 gaps+=run['inter_epoch_cycles']
maximum=max(x['low_section_end']+x['stack_allowance_bytes'] for x in sram['records'])
out=dict(physical=r['physical'],full_model=False,full_input_width=False,full_output_rows=False,application=r['application'],worker_pes=6,controller_pes=1,passive_pes=1,fp8_slots_per_worker=112,bf16_slots_per_worker=12,
 max_compiled_sram_plus_declared_stack=maximum,sram_margin_bytes=48128-maximum,request_to_result_cycles={k:dict(min=min(v),median=statistics.median(v),max=max(v)) for k,v in groups.items()},inter_epoch_cycles=dict(min=min(gaps),median=statistics.median(gaps),max=max(gaps)),
 total_cycles_per_96epoch_run=[x['autonomous_total_cycles'] for x in r['runs']],cycles_per_epoch_including_controller_gaps=[x['cycles_per_epoch'] for x in r['runs']],epochs=192,local_and_subtree_fp32_values_checked=192*6*4,weight_uploads=1,
 max_abs_root_fp64_error=max(x['max_abs_root_fp64_error'] for x in r['runs']),fixture_sha256=r['fixture_sha256'],scope=r['scope'],timing_note=r['timing_note'],clock_hz=None,
 interpretation='Device control depends on prior result completion. Independent preloaded neural inputs are not an autoregressive sequence. Full-model spatial ownership, complete layers and token rate remain unqualified. PerPE192-byte oracle is validation instrumentation included in SRAM and timing.')
with a.output.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
