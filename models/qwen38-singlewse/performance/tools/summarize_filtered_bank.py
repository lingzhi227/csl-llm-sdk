"""Produce a bounded component receipt without deriving model throughput."""
import argparse,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=ROOT/'evidence'/a.name;r=json.loads((root/'result.json').read_text());complete=json.loads((root/'COMPLETE.json').read_text())
assert r['passed'] and r['normal_stop'] and r['physical'] and not r['full_model']
assert complete['all_owned_jobs_released'] and all(s['exit_code']==0 and not s['cleanup_errors'] and all(j['released'] and j['phase']=='SUCCEEDED' for j in s['jobs']) for s in complete['stages'])
fixture=json.loads((root/'fixture.json').read_text());assert r['fixture_sha256']==fixture['fixture_sha256']
assert [c['case'] for c in r['cases']]==fixture['physical_cases']==list(range(10))
assert r['all_weights_retained'] and r['counter_windows_exact'] and r['all_packet_and_teardown_counters_exact']
sram=json.loads((root/'sram.json').read_text());assert sram['passed'] and sram['application_pes']==20
maximum=max(t['low_section_end']+t['stack_allowance_bytes'] for t in sram['records'])
timing={}
for kind in ['fp8','bf16']:
 values=[v for c in r['cases'] if c['kind']==kind for v in c['received_to_dot_done_cycles']]
 timing[kind]=dict(samples=len(values),min=min(values),median=statistics.median(values),max=max(values))
storage=sum(w['fp8_slots']*260+w['bf16_slots']*512 for w in fixture['workers'])
result=dict(schema='wse-filtered-bank-qualification-v1',physical=True,full_model=False,attempt=a.name,application=[4,5],
 original_model=fixture['model'],original_revision=fixture['revision'],fixture_sha256=fixture['fixture_sha256'],
 epochs=10,native_dot_values=10*12*2,exact_worker_packet_words=10*12*65,exact_relay_pair_words=10*3*130,
 full_original_bank_readback_bytes=storage,full_transfer_readback_bytes=r['bank_transfer_bytes'],retained_padding_bytes=r['retained_padding_bytes'],maximum_bank_payload_bytes=35256,max_compiled_sram_plus_stack=maximum,
 sram_margin_bytes=48128-maximum,all_original_weights_retained=True,counter_windows_exact=True,all_jobs_released=True,
 windows=dict(horizontal_periods=[260,780],horizontal_counts=[130],horizontal_offsets=[0,130,260,390,520,650],vertical_period=130,vertical_count=65,vertical_offsets=[0,65]),
 local_received_to_dot_done_cycles=timing,timing_note=r['timing_note'],clock_hz=None,full_model_speed_target_achieved=False,
 artifact_sha256=json.loads((root/'artifact.json').read_text())['sha256'],scope=r['scope'])
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
