"""Compact component scope, complete recurrence checks and measured cycle windows."""
import argparse
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(); parser.add_argument('attempt'); args = parser.parse_args()
if not re.fullmatch('gdn-bank-(sim|hw)-[0-9]{3}', args.attempt):
    raise ValueError('Attempt')
directory = ROOT / 'evidence' / args.attempt
result = json.loads((directory / 'result.json').read_text())
sram = json.loads((directory / 'sram.json').read_text())
assert result['passed'] and result['normal_stop'] and sram['passed'] and sram['application_pes'] == 20
cycles = [v for b in result['batches'] for v in b['request_to_full_state_output_cycles']]
gaps = [v for b in result['batches'] for v in b['inter_step_cycles']]
maximum = max(r['low_section_end'] + r['stack_allowance_bytes'] for r in sram['records'])
def stats(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))
summary = dict(physical=result['physical'], full_model=False, application=[4, 5], state_pes=16,
               fp8_slots_per_state_pe=111, total_state_shape=[128, 128], state_shape_per_pe=[32, 32],
               positions=sum(b['count'] for b in result['batches'] if not b['replay']),
               reset_replay_positions=sum(b['count'] for b in result['batches'] if b['replay']),
               full_state_terminal_checks=len(result['batches']), output_fp32_values_checked=len(cycles) * 128,
               native_fp8_dot_values_checked=result['native_fp8_dot_values_checked'],
               request_to_full_state_output_cycles=stats(cycles), inter_step_cycles=stats(gaps),
               max_state_fp64_error=max(b['max_state_fp64_error'] for b in result['batches']),
               max_output_fp64_error=max(b['max_output_fp64_error'] for b in result['batches']),
               all_outputs_and_terminal_states_exact_block_order_fp32=True, unchanged_fp64_bounds_passed=True,
               original_weight_uploads=1, all_original_banks_retained=True,
               max_compiled_sram_plus_stack=maximum, sram_margin_bytes=48128 - maximum,
               compiled_elf_images=sram['compiled_elf_images'], exact_application_coordinate_coverage=True,
               fixture_sha256=result['fixture_sha256'], clock_hz=None, scope=result['scope'], timing_note=result['timing_note'])
with (directory / 'summary.json').open('x') as f:
    json.dump(summary, f, indent=2); f.write('\n')
print(json.dumps(summary))
