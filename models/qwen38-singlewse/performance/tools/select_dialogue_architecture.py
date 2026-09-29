"""Materialize the selected design and its larger-context byte reservation.

Selection is an engineering decision, not execution acceptance. Preserve the
source screen; reserve exact KV pages and explicit controller cohosts before
downstream lowering. No model payload, compiler or remote allocation is used.
"""
import hashlib,json,sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_schedule import rank_xy,audit_native_schedule
from prediction.placement_paths import route_handoffs


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh immutable selection directory required')
    source=ROOT/'performance/evidence/architecture-screen-010/native-plan.json';raw=source.read_bytes()
    plan=json.loads(raw);old_context=plan['config']['context'];context=512
    assert old_context==96 and plan['config']['concurrency']==1
    plan['config']['context']=context;state_added=0;controllers=[];budgets=[];kv_regions=0
    for stage in plan['stages']:
        candidates=[]
        for region in stage['regions']:
            kv={n['attributes']['state']for n in region['nodes']if n['op']=='kv_append'}
            for node in region['nodes']:
                if node['op']=='kv_append':node['attributes']['context']=context
            cursor=0
            for item in region['auxiliary']:
                if item['id']in kv:
                    per_position=item['bytes_per_request']//old_context
                    state_added+=(context-old_context)*per_position;kv_regions+=1
                    item['bytes_per_request']=item['bytes']=context*per_position
                    item['pages']=(item['bytes']+127)//128
                item['page_start']=cursor;cursor+=item['pages']
            region['auxiliary_pages']=cursor;region['banks']['auxiliary_pages']=cursor
            if cursor>region['banks']['auxiliary_page_capacity']:raise ValueError('Context expansion exceeds a region')
            actor_ranks={v['rank']for v in region.get('quantization_actors',[])}
            maximum=0;minimum=48128
            for cls in region['banks']['classes']:
                for rank in range(cls['rank_start'],cls['rank_end']):
                    prefix=cls['page_prefix']+(rank-cls['rank_start'])*cls['auxiliary_pages']
                    live=min(cls['auxiliary_pages'],max(0,cursor-prefix))
                    payload=cls['fp8_slots']*260+cls['bf16_slots']*256+live*128
                    maximum=max(maximum,payload);minimum=min(minimum,payload)
                    if rank not in actor_ranks and payload<=28000:
                        candidates.append(dict(region=region['id'],rank=rank,pe=rank_xy(region,rank),payload_bytes=payload))
            budgets.append(dict(stage=stage['id'],region=region['id'],role=region['role'],rect=region['rect'],
                maximum_actual_payload_bytes=maximum,minimum_actual_payload_bytes=minimum,
                stack_allowance_bytes=4096,code_sdk_allowance_bytes=7424,transport_scratch_allowance_bytes=1352,
                minimum_planned_margin_bytes=48128-maximum-4096-7424-1352,
                spare_auxiliary_pages=region['banks']['auxiliary_page_capacity']-cursor,
                compiled_role_sram=False))
        if not candidates:raise ValueError('No explicitly budgeted stage controller candidate: '+stage['id'])
        controller=min(candidates,key=lambda c:(c['payload_bytes'],c['pe']))
        controller.update(stage=stage['id'],mode='dedicated'if controller['payload_bytes']==0 else'cohost',
                          extra_control_code_state_budget_bytes=3072,
                          minimum_planned_margin_bytes=48128-controller['payload_bytes']-4096-7424-1352-3072,
                          source_status='Coordinate/budget only; arrival-driven controller and cohost leases require implementation')
        assert controller['minimum_planned_margin_bytes']>=0
        controllers.append(controller);stage['planned_controller']=controller
    assert kv_regions==16
    plan['state_bytes_per_request']+=state_added
    plan.update(schema='wse-dialogue-architecture-selection-v1',selected_for_implementation=True,
                selection_source_native_plan_sha256=hashlib.sha256(raw).hexdigest(),
                controller_placement_status='66 explicit dedicated/cohost budgets; runtime leases unqualified',
                neural_executable=False,physical=False,compiled_sram_qualified=False,
                stale_capacity_audit_removed=True)
    plan.pop('audit',None)
    audit=audit_native_schedule(plan,json.loads((ROOT/'configs/tensors.json').read_text())['tensors'],require_controller=False)
    paths=route_handoffs(plan,'owner_aligned')
    report=dict(selected=True,source_screen='architecture-screen-010',context=context,context_added_state_bytes=state_added,
                state_bytes_per_request=plan['state_bytes_per_request'],native_bank_audit=audit,
                controller_counts=dict(Counter(c['mode']for c in controllers)),
                controller_placement_budget_checked=True,controller_runtime_qualified=False,
                controllers=controllers,region_budgets=budgets,handoff_paths=paths,
                whole_model_executed=False,physical=False,tps=None,
                scope='Exact native banks/512-position KV allocation, planned control placement and handoff geometry. '
                      'GDN head-local remap, complete fabric/resource leases, compiled SRAM and neural/runtime performance remain unqualified.')
    a.output.mkdir()
    for name,obj in [('selected-plan.json',plan),('selection.json',report)]:
        (a.output/name).write_text(json.dumps(obj,separators=(',',':'))+'\n')
    sources=[Path(__file__),ROOT/'performance/spatial/layer_schedule.py',ROOT/'performance/prediction/placement_paths.py',source]
    manifest={}
    for path in sources:
        rel=str(path.relative_to(ROOT));data=path.read_bytes();target=a.output/'source'/rel
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);manifest[rel]=hashlib.sha256(data).hexdigest()
    (a.output/'source-manifest.json').write_text(json.dumps(dict(files=manifest),indent=2)+'\n')
    print(json.dumps(dict(selected=True,context=context,state_bytes_per_request=plan['state_bytes_per_request'],
                         bank_audit=audit,controllers=report['controller_counts'],physical=False,tps=None)))


if __name__=='__main__':main()
