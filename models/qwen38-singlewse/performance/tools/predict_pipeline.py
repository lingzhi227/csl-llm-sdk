"""Screen emitted spatial plans and optionally simulate sourced service costs.

Read-only inputs, fresh output directory, no remote jobs and no model payloads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'performance'))
from prediction.model import read_json,analyze,service_template,evaluate_scenario


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--ir',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,default=ROOT/'performance/evidence/native-shapes-summary-001.json')
    parser.add_argument('--service-profile',type=Path)
    parser.add_argument('--concurrency',type=int)
    parser.add_argument('--target-tps',type=float,default=2000)
    parser.add_argument('--scalar-scales',action='store_true')
    parser.add_argument('--warmup',type=int,default=100)
    parser.add_argument('--samples',type=int,default=1000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory; existing results are never overwritten')
    if args.concurrency is not None and args.service_profile is None:
        parser.error('--concurrency requires --service-profile; changing resident capacity requires a new plan')
    manifest={}
    def load(path):
        obj,digest=read_json(path);manifest[str(path.resolve())]=digest;return obj,digest
    plan,plan_hash=load(args.plan);ir,_=load(args.ir);calibration,_=load(args.calibration)
    report=analyze(plan,ir,calibration,target_tps=args.target_tps,vector_scales=not args.scalar_scales)
    if args.service_profile:
        scenario,_=load(args.service_profile)
        report['conditional_service_scenario']=evaluate_scenario(plan,plan_hash,scenario,
            concurrency=args.concurrency,warmup=args.warmup,samples=args.samples)
    files=[ROOT/'performance/prediction'/n for n in ['__init__.py','model.py','schedule.py']]+[Path(__file__).resolve()]
    report['input_sha256']=manifest
    report['implementation_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    args.output.mkdir(parents=True)
    for name,obj in [('report.json',report),('service-profile-template.json',service_template(plan,plan_hash))]:
        with (args.output/name).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
    lines=['# Pipeline performance screening','',report['scope'],'',
           f"- Resident concurrency: {report['concurrency']}; context: {report['context']}.",
           f"- Target: {args.target_tps:g} aggregate generated tokens/s.",
           f"- Necessary mean request cycle at that concurrency: {report['requirements']['maximum_mean_request_feedback_cycle_seconds']*1000:g} ms.",
           f"- Matching native-shape MAC coverage: {report['coverage']['calibrated_shape_mac_fraction']:.2%}. This is not timing coverage.",
           '- Full-model TPS prediction: unavailable (unmeasured components and missing clock calibration).','',
           '## Conditional native-work bottlenecks','',
           '| Region | PEs | Median native busy cycles |','|---|---:|---:|']
    lines += [f"| {r['id']} | {r['pes']} | {r['cycles']['median']:.2f} |" for r in report['top_native_work_regions']]
    lines += ['','These reuse measured tile costs; they exclude communication and other operators.',
              'Observed sample min/max are not prediction confidence intervals.','','## Missing costs','']
    lines += ['- '+s for s in report['unknowns']]
    if 'conditional_service_scenario' in report:
        s=report['conditional_service_scenario']
        lines += ['','## Explicit service scenario','',
                  f"Simulated completions/cycle: {s['generated_completions_per_cycle']:.8g}.",
                  f"Conditional tokens/s: {s['generated_tokens_per_second']} (clock source: {s['clock_source']}).",
                  'This is a supplied-parameter queue simulation, not hardware acceptance.']
    with (args.output/'REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(report=str(args.output/'REPORT.md'),
                         complete_model_prediction=False,
                         calibrated_shape_mac_fraction=report['coverage']['calibrated_shape_mac_fraction'],
                         required_mean_request_cycle_ms=report['requirements']['maximum_mean_request_feedback_cycle_seconds']*1000,
                         unknowns=len(report['unknowns']))))


if __name__=='__main__':main()
