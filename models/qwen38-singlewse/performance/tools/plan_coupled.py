"""Bind all original projection descriptors and forests to the columnar overlay."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.columnar import materialize
from spatial.projections import bundle_graph
from spatial.forest import document
p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--overlay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('Frozen plan exists')
atlas=materialize(a.atlas,a.overlay);raw=(json.dumps(atlas,indent=2)+'\n').encode()
graph_path=ROOT.parent/'configs/model-graph.json';graph_raw=graph_path.read_bytes();graph=json.loads(graph_raw)
assert hashlib.sha256(graph_raw).hexdigest()==atlas['provenance']['model-graph.json']
bundles=bundle_graph(graph,atlas);bundles['provenance']=dict(atlas_sha256=hashlib.sha256(raw).hexdigest(),graph_sha256=hashlib.sha256(graph_raw).hexdigest(),base_atlas_sha256=atlas['columnar_base_atlas_sha256'],overlay_sha256=atlas['columnar_overlay_sha256'])
forests=document(atlas,bundles);forests['columnar_overlay_sha256']=atlas['columnar_overlay_sha256']
a.output.mkdir()
for name,data in [('bundles.json',bundles),('forests.json',forests)]:
 with (a.output/name).open('x') as f:json.dump(data,f,indent=2);f.write('\n')
print(json.dumps(bundles['metrics']))
