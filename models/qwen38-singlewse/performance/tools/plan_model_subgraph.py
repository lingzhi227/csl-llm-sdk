"""Bind the full-dimensional real MLP planning boundary to pinned source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.model_subgraph import mlp_subgraph
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
g=ROOT/'configs/model-graph.json';t=ROOT/'configs/tensors.json';graph=json.loads(g.read_text());tensors=json.loads(t.read_text());assert graph['revision']==tensors['revision']
d=mlp_subgraph(graph,tensors['tensors'],a.layer)
d['source_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [g,t]}
with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
print(json.dumps(dict(nodes=[n['id'] for n in d['original_nodes']],resources=d['resource_counts'],executable=d['executable'],output=str(a.output))))
