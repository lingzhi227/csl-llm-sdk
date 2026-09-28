"""Compare accepted complete-MLP controller counters at fixed oracle identity.

Raw cycle ratios are not calibrated elapsed-time ratios or generated-token TPS.
The P27 baseline has no explicit completed-output wall-time boundary, so no wall
speedup can be inferred from that pair. Preserve every case, including zero.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def compare(before,after):
    required=['passed','physical','normal_stop','original_weights_resident','all_original_banks_retained',
              'all_luts_retained','all_native_inputs_exact','all_counters_exact','all_pe_drained']
    for result in [before,after]:
        if not all(result[k] is True for k in required) or result['full_model'] is not False:
            raise ValueError('Accepted complete component results required')
        if result['original_output_values_checked']!=20480 or len(result['epochs'])!=4:
            raise ValueError('Complete four-case qualification required')
    if before['fixture_metadata_sha256']!=after['fixture_metadata_sha256']:raise ValueError('Different frozen oracle')
    cases=[]
    for old,new in zip(before['epochs'],after['epochs']):
        if (old['index'],old['epoch'])!=(new['index'],new['epoch']):raise ValueError('Different workload order')
        if not all(old[k] is True and new[k] is True for k in ['all_outputs_bit_exact','all_native_inputs_exact','all_counters_exact','all_pe_drained']):
            raise ValueError('Unqualified case')
        a=old['controller_cycles'];b=new['controller_cycles']
        if not a>0 or not b>0:raise ValueError('Invalid counters')
        phases=new['controller_phase_cycles']
        if any(v<=0 for v in phases.values()) or sum(phases.values())!=b:raise ValueError('Invalid phase decomposition')
        if new['completed_output_seconds']<new['host_arm_start_finish_seconds']:raise ValueError('Invalid completed-output timer')
        cases.append(dict(index=new['index'],epoch=new['epoch'],baseline_cycles=a,candidate_cycles=b,
            baseline_to_candidate_cycle_ratio=a/b,cycle_reduction_fraction=1-b/a,
            candidate_completed_output_seconds=new['completed_output_seconds'],candidate_phase_cycles=phases))
    return dict(passed=True,original_output_values_checked=20480,fixture_metadata_sha256=after['fixture_metadata_sha256'],
        cases=cases,cycle_to_seconds_calibration_performed=False,wall_time_speedup=None,full_model_tps=None,
        note='Phase counters mark source distribution completion and final receipt, not isolated kernel durations; neural work overlaps these intervals. Baseline lacks completed-output wall time. No aggregate model speed acceptance.')


def main():
    p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    records=[];bindings={}
    for name in [args.baseline,args.candidate]:
        root=ROOT/'evidence'/name;complete=json.loads((root/'COMPLETE.json').read_text())
        if not complete['all_owned_jobs_released']:raise ValueError('Attempt not released')
        raw=(root/'result.json').read_bytes();records.append(json.loads(raw))
        bindings[name]={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['result.json','source-manifest.json','COMPLETE.json','sram-summary.json']}
    result=dict(baseline=args.baseline,candidate=args.candidate,source_bindings=bindings,**compare(*records))
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
