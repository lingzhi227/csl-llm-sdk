"""Freeze a full-dimensional spatial/fusion candidate and independent audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.model_subgraph import mlp_subgraph
from spatial.mlp_regions import MlpRegions,MlpRegionSpec
from tools.audit_mlp_regions import audit
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--columns',type=int,default=9);p.add_argument('--chunk-words',type=int,default=8);a=p.parse_args()
g=ROOT/'configs/model-graph.json';t=ROOT/'configs/tensors.json'
graph=json.loads(g.read_text());tensors=json.loads(t.read_text());assert graph['revision']==tensors['revision']
flow=MlpRegions(mlp_subgraph(graph,tensors['tensors']),MlpRegionSpec(columns=a.columns,down_chunk_words=a.chunk_words))
d=flow.document();d['audit']=audit(d,flow.original_tiles(),flow.native_edges())
paths=[g,t]+[ROOT/'performance'/n for n in ['spatial/mlp_regions.py','spatial/mlp_fusion.py','spatial/model_subgraph.py','spatial/region_epoch.py','tools/audit_mlp_regions.py','protocols/region_epoch.csl','csl/mlp_fused.csl']]+[ROOT/'csl/qwen_math.csl']
d['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(dict(application=d['application'],ownership=d['ownership'],audit=d['audit'],executable=False)))
