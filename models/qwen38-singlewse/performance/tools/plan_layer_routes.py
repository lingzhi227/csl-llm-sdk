"""Freeze complete native MLP tree audit, selected endpoints and exact source.

No device launch. Unlowered ingress, egress and external credits remain explicit.
The pinned resident schedule remains in its existing frozen evidence directory.
"""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_routes import audit_mlp_network,contraction_groups,fused_deliveries


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output directory required')
    inputs=['performance/evidence/layer-native-schedule-002/layer-schedule.json',
            'performance/evidence/layer-projection-compile-007/source-manifest.json',
            'performance/evidence/layer-backend-compile-009/source-manifest.json']
    plan=json.loads((ROOT/inputs[0]).read_text());audit=audit_mlp_network(plan)
    sources=['performance/spatial/layer_routes.py','performance/spatial/layer_schedule.py',
             'performance/runtime/layer_projection.csl','performance/csl/layer_native.csl',
             'performance/csl/layer_fusion.csl','performance/runtime/layer_backend.csl',
             'performance/tests/test_layer_routes.py','performance/tools/plan_layer_routes.py']
    a.output.mkdir(parents=True)
    def write(name,value):
        with (a.output/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write('audit.json',audit)
    groups=[];deliveries=[]
    for stage in plan['stages'][1:3]:
        for region in stage['regions']:
            if region['role'] not in ('gate_up','down'):continue
            for g in contraction_groups(region):groups.append({k:v for k,v in g.items() if k!='nodes'})
            if region['role']=='gate_up':deliveries.extend(fused_deliveries(region))
    write('selected-layer0-1.json',dict(groups=groups,projection_deliveries=deliveries,
          reduction_order='Local original K slices in increasing index, then left subtree, then right subtree; BF16 only at root.',
          input_packet='u32 epoch, u32 local part, columns/2 native FP16 words, u32 original activation scale',
          reduction_packet='u32 epoch, u32 row iteration, u32 original K contributors, branch-major FP32 partials',
          root_packet='u32 epoch, u32 row iteration, u32 first original output row, branch-major packed BF16 values',
          consumer_credit='u32 epoch, u32 row iteration; copied projection block credits before whole quantization group completion',
          complete_layer_connected=False))
    manifest=dict(files={},inputs={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in inputs})
    for n in sources:
        raw=(ROOT/n).read_bytes();manifest['files'][n]=hashlib.sha256(raw).hexdigest()
        dst=a.output/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
    write('source-manifest.json',manifest);print(json.dumps(audit))


if __name__=='__main__':main()
