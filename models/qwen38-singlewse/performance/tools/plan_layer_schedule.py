"""Freeze a full resident native-loop address schedule and its selected layers."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.layer_schedule import lower_layers,audit_native_schedule


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh frozen output directory required')
    paths=['performance/evidence/pipeline-stage-map-002/stage-map.json','performance/evidence/native-shapes-hw-002/result.json',
           'configs/tensors.json','performance/spatial/layer_schedule.py','performance/runtime/layer_weights.py',
           'performance/csl/layer_native.csl','performance/csl/layer_fusion.csl','performance/csl/gdn_page.csl','performance/csl/bf16_row.csl',
           'performance/csl/pipeline_lease.csl','performance/runtime/layer_backend.csl','performance/runtime/layer_controller.csl',
           'performance/tools/plan_layer_schedule.py']
    source={n:(ROOT/n).read_bytes() for n in paths}
    plan=lower_layers(json.loads(source[paths[0]]),json.loads(source[paths[1]]))
    plan['audit']=audit_native_schedule(plan,json.loads(source[paths[2]])['tensors'])
    a.output.mkdir(parents=True)
    def write(name,value):
        with (a.output/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write('layer-schedule.json',plan)
    write('selected-layer0-1.json',dict(stages=plan['stages'][1:3],config=plan['config'],
          source_schedule='layer-schedule.json',neural_execution=False,complete_fabric_routes=False))
    write('source-manifest.json',dict(files={n:hashlib.sha256(b).hexdigest() for n,b in source.items()}))
    for n,b in source.items():
        dest=a.output/'source'/n;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(b)
    print(json.dumps(plan['audit']))


if __name__=='__main__':main()
