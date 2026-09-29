"""Bounded front-loaded whole-model candidate comparison; no compiler or jobs.

Capacity/provenance cover the complete original model. Network measurements in
this report are explicit geometric counts of one complete handoff subgraph;
unknown kernels, state access, reductions, I/O and scheduling remain unknown.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib,json,signal,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.pipeline import PipelineConfig,place_pipeline,audit_pipeline
from spatial.pipeline_lowering import lower_pipeline
from spatial.layer_schedule import lower_layers
from prediction.placement_paths import route_handoffs
from prediction.model import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--columns',type=int,choices=[4,8,16],required=True)
    p.add_argument('--mirror',action='store_true',help='Reflect each internal T partition along both stage axes')
    p.add_argument('--seconds',type=int,default=180)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists()or not 1<=a.seconds<=300:p.error('Fresh output and 1..300 seconds required')
    a.output.mkdir();begin=time.monotonic()
    def expired(*_):raise TimeoutError('Bounded architecture screen reached its limit')
    signal.signal(signal.SIGALRM,expired);signal.alarm(a.seconds)
    paths=['configs/model-graph.json','configs/tensors.json','performance/evidence/native-shapes-summary-001.json',
           'performance/spatial/pipeline.py','performance/spatial/layer_schedule.py','performance/spatial/layer_routes.py',
           'performance/spatial/pipeline_lowering.py','performance/prediction/placement_paths.py',
           'performance/prediction/model.py','performance/tools/compare_pipeline_architectures.py']
    frozen={n:(ROOT/n).read_bytes()for n in paths}
    manifest={n:hashlib.sha256(v).hexdigest()for n,v in frozen.items()}
    try:
        graph=json.loads(frozen[paths[0]]);tensors=json.loads(frozen[paths[1]])['tensors'];cal=json.loads(frozen[paths[2]])
        config=PipelineConfig(macro_columns=a.columns,concurrency=1)
        plan=place_pipeline(graph,tensors,config)
        if a.mirror and plan['placement_admitted']:
            for stage in plan['stages']:
                sx,sy,sw,sh=stage['rect']
                for r in stage['regions']:
                    x,y,w,h=r['rect'];r['rect']=[sx+sw-(x-sx)-w,sy+sh-(y-sy)-h,w,h]
        plan['audit']=audit_pipeline(plan,graph,tensors)
        print(json.dumps(dict(phase='capacity',columns=a.columns,mirror=a.mirror,audit=plan['audit'])),flush=True)
        report=dict(passed=True,columns=a.columns,mirror=a.mirror,capacity=plan['audit'],
                    required_width=plan['required_width'],configuration=vars(config),
                    compiled=False,physical=False,full_model_rate=None,architecture_selected=False)
        if plan['placement_admitted']:
            ir=lower_pipeline(plan,graph,tensors)
            report['partial_native_screen']=analyze(plan,ir,cal)
            native=lower_layers(plan,cal)
            report['native_addressing_admitted']=native['native_loop_addressing_admitted']
            report['native_address_failures']=native['failed_regions']
            report['original_operations']=dict(Counter(n['op']for n in graph['nodes']))
            report['stage_shapes']=dict(Counter(str(s['rect'][2:])for s in plan['stages'][1:-1]))
            report['state_bytes_per_request']=plan['state_bytes_per_request']
            # Context extension accounting only: unchanged weight banks and every
            # other auxiliary reservation. All attention regions must have room.
            context_limits=[]
            for s in native['stages']:
                for r in s['regions']:
                    kv=[n for n in r['nodes']if n['op']=='kv_append']
                    if kv:
                        st=next(v for v in r['auxiliary']if v['id']==kv[0]['attributes']['state'])
                        per_position=st['bytes_per_request']//config.context
                        spare=r['banks']['auxiliary_page_capacity']-r['auxiliary_pages']
                        context_limits.append(config.context+max(0,spare)*128//per_position)
            report['unchanged_region_context_byte_ceiling']=min(context_limits)
            report['context_scope']='Byte-capacity ceiling only, not extended context allocation, numerical or timing qualification'
            report['handoff_paths']=[route_handoffs(native,policy)for policy in ('fixed_lane','owner_aligned')]
            report['io']=dict(initial_original_weight_bytes=plan['audit']['original_weight_bytes'],
                              prompt_token_id_bytes_per_token=4,output_token_id_bytes_per_token=4,
                              full_bf16_logits_bytes_per_token=248320*2,
                              token_id_payload_bytes_per_second_at_target=2000*4,
                              measured_sdk_endpoint_mapping=None,small_message_latency_seconds=None,
                              prefill_device_latency_seconds=None,
                              scope='Declared resident token-ID interface; actual framing, template, channels and control traffic unmeasured')
            report['unmeasured_decision_inputs']=[
                'GDN and attention state locality, live context-dependent service times and resource leases',
                'Native reduction/epilogue and RMS/quantization readiness on the critical path',
                'Gate/up fused activation and down fanout paths and shared physical links',
                'Embedding, full-vocabulary head selection and dependent feedback route',
                'Composed ELF SRAM, color/queue/DSR/microthread admission and backpressure',
                'Initial load, append prefill, first/last output, SDK endpoint/control latency']
            report['decision']=('Keep these as bounded candidates. No topology wins from capacity or this handoff subgraph alone. '
                                'Use full dependent-path service evidence before migration; do not expand fixed-coordinate integration by default.')
            (a.output/'native-plan.json').write_text(json.dumps(native,separators=(',',':'))+'\n')
        (a.output/'stage-map.json').write_text(json.dumps(plan,separators=(',',':'))+'\n')
    except (TimeoutError,ValueError)as error:
        report=dict(passed=False,columns=a.columns,mirror=a.mirror,error=type(error).__name__,message=str(error),
                    physical=False,compiled=False,architecture_selected=False)
    finally:
        signal.alarm(0)
    report.update(wall_seconds=time.monotonic()-begin,source_sha256=manifest)
    for n,raw in frozen.items():
        target=a.output/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('partial_native_screen','handoff_paths','source_sha256')}),flush=True)


if __name__=='__main__':main()
