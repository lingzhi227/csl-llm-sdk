"""Summarize exact composed-engine and exhaustive decoder physical evidence."""
import argparse,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=ROOT/'evidence'/a.name;r=json.loads((root/'result.json').read_text());complete=json.loads((root/'COMPLETE.json').read_text())
assert r['passed'] and r['normal_stop'] and r['physical'] and not r['full_model']
assert complete['all_owned_jobs_released'] and all(s['exit_code']==0 and not s['cleanup_errors'] and all(j['released'] and j['phase']=='SUCCEEDED' for j in s['jobs']) for s in complete['stages'])
fixture=json.loads((root/'fixture.json').read_text());assert r['fixture_sha256']==fixture['fixture_sha256']
assert [c['case'] for c in r['cases']]==fixture['physical_cases']==list(range(10))
for gate in ['all_weights_retained','counter_windows_exact','all_packet_and_teardown_counters_exact','all_subtrees_exact','controller_result_exact']:assert r[gate]
decoder=r['exhaustive_decoder'];assert decoder['packed_patterns']==65536 and decoder['mismatches']==0
old=sum(decoder['reference_decode_cycles']);new=sum(decoder['shift_decode_cycles']);assert old>0 and new>0
sram=json.loads((root/'sram.json').read_text());assert sram['passed'] and sram['application_pes']==20
maximum=max(t['low_section_end']+t['stack_allowance_bytes'] for t in sram['records'])
cycles=[c['source_send_to_result_return_cycles'] for c in r['cases']]
result=dict(schema='wse-coupled-bank-qualification-v1',physical=True,full_model=False,attempt=a.name,application=[4,5],
 original_model=fixture['model'],original_revision=fixture['revision'],fixture_sha256=fixture['fixture_sha256'],
 epochs=10,native_dot_values=240,subtree_values=240,controller_values=20,exact_worker_packet_words=7800,exact_relay_pair_words=3900,
 full_original_bank_readback_bytes=sum(w['fp8_slots']*260+w['bf16_slots']*512 for w in fixture['workers']),
 full_transfer_readback_bytes=r['bank_transfer_bytes'],retained_padding_bytes=r['retained_padding_bytes'],maximum_bank_payload_bytes=35256,
 max_compiled_sram_plus_stack=maximum,sram_margin_bytes=48128-maximum,all_original_weights_retained=True,
 all_subtrees_and_controller_results_exact=True,counter_windows_exact=True,all_jobs_released=True,
 exhaustive_decoder=dict(**decoder,reference_total_cycles=old,shift_total_cycles=new,matched_decoder_speedup=old/new),
 source_send_to_result_return_cycles=dict(min=min(cycles),median=statistics.median(cycles),max=max(cycles),cases=[dict(case=c['case'],kind=c['kind'],mode=c['mode'],period=c['period_words'],cycles=c['source_send_to_result_return_cycles']) for c in r['cases']]),
 timing_note=r['timing_note'],clock_hz=None,full_model_speed_target_achieved=False,
 artifact_sha256=json.loads((root/'artifact.json').read_text())['sha256'],scope=r['scope'])
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
