"""Freeze complete-dimensional static MLP routes and independent fabric audit."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.model_subgraph import mlp_subgraph
from spatial.mlp_regions import MlpRegions
from spatial.mlp_static import MlpStatic
from tools.audit_mlp_static import audit
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
graph=json.loads((ROOT/'configs/model-graph.json').read_text());tensors=json.loads((ROOT/'configs/tensors.json').read_text())['tensors']
fabric=MlpStatic(MlpRegions(mlp_subgraph(graph,tensors)));result=fabric.document();result['audit']=audit(fabric)
paths=['configs/model-graph.json','configs/tensors.json']+['performance/'+s for s in ['spatial/mlp_static.py','spatial/mlp_regions.py','spatial/contraction.py','spatial/model_subgraph.py','tools/audit_mlp_static.py']]
result['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
with a.output.open('x') as out:json.dump(result,out,indent=2);out.write('\n')
print(json.dumps(result['audit']))
