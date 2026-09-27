"""Freeze exact full-model metadata residency and independent interval audit."""
import argparse,hashlib,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.model_subgraph import mlp_subgraph
from spatial.mlp_regions import MlpRegions
from spatial.mlp_residency import MlpResidency
from tools.audit_mlp_residency import audit
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
g=ROOT/'configs/model-graph.json';t=ROOT/'configs/tensors.json';graph=json.loads(g.read_text());header=json.loads(t.read_text())
assert graph['revision']==header['revision'];tensors=header['tensors']
r=MlpResidency(MlpRegions(mlp_subgraph(graph,tensors)),graph,tensors);d=r.document();d['audit']=audit(r,graph,tensors)
paths=[g,t]+[ROOT/'performance'/n for n in ['spatial/model_subgraph.py','spatial/mlp_regions.py','spatial/mlp_residency.py','tools/audit_mlp_residency.py']]
d['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(d['audit']))
