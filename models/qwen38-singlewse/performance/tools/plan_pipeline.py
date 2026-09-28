"""Emit the full resident stage map and layer-local kernel/stream IR; no jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.pipeline import PipelineConfig,place_pipeline,audit_pipeline
from spatial.pipeline_lowering import lower_pipeline,audit_boundaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=2);a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh immutable output directory required')
    graph=json.loads((ROOT/'configs/model-graph.json').read_text())
    tensors=json.loads((ROOT/'configs/tensors.json').read_text())['tensors']
    plan=place_pipeline(graph,tensors,PipelineConfig(concurrency=a.concurrency))
    plan['audit']=audit_pipeline(plan,graph,tensors)
    ir=lower_pipeline(plan,graph,tensors) if plan['placement_admitted'] else None
    if ir:ir['audit']=audit_boundaries(ir)
    sources=['configs/model-graph.json','configs/tensors.json']+[
        'performance/'+n for n in ['spatial/pipeline.py','spatial/pipeline_lowering.py',
                                   'spatial/pipeline_protocol.py','spatial/model_subgraph.py','csl/pipeline_lease.csl','tools/plan_pipeline.py']]
    manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
    a.output.mkdir(parents=True)
    def write(name,value):
        with (a.output/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    write('stage-map.json',plan);write('source-manifest.json',dict(files=manifest))
    for name,digest in manifest.items():
        raw=(ROOT/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Source changed during freeze')
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(raw)
    if ir:write('layer-kernel-ir.json',ir)
    sweep=[]
    for count in (1,2,3,4,8,16,32,64):
        candidate=place_pipeline(graph,tensors,PipelineConfig(concurrency=count))
        sweep.append(dict(concurrency=count,required_width=candidate['required_width'],
                          state_bytes=count*candidate['state_bytes_per_request'],
                          placement_admitted=candidate['placement_admitted'],
                          geometry_scope='This packing/configuration only, not a global impossibility proof'))
    write('capacity-sweep.json',sweep)
    print(json.dumps(dict(placement=plan['audit'],streams=ir['audit'] if ir else None)))


if __name__=='__main__':main()
