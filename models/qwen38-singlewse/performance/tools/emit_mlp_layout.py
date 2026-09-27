"""Freeze candidate full-wafer MLP composition without compiling or reserving it."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from spatial.model_subgraph import mlp_subgraph
from spatial.mlp_regions import MlpRegions
from spatial.mlp_residency import MlpResidency
from spatial.mlp_layout import emit
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
graph=json.loads((ROOT/'configs/model-graph.json').read_text());tensors=json.loads((ROOT/'configs/tensors.json').read_text())['tensors']
r=MlpResidency(MlpRegions(mlp_subgraph(graph,tensors)),graph,tensors)
assert [(n['slot'],n['tensor']) for n in r.norms]==[(2*layer+offset,f'model.language_model.layers.{layer}.{kind}.weight') for layer in range(64) for offset,kind in enumerate(['input_layernorm','post_attention_layernorm'])]+[(128,'model.language_model.norm.weight')]
files={'layout.csl':emit(r).encode()}
origins={}
for name,path in {
 **{n+'.csl':'performance/runtime/'+n+'.csl' for n in ['mlp_worker','mlp_activation','mlp_join','mlp_norm','mlp_idle']},
 'fp8_shape.csl':'performance/probes/native_shapes/fp8_shape.csl','fp8_unpack_shift.csl':'performance/csl/fp8_unpack_shift.csl',
 'qwen_math.csl':'csl/qwen_math.csl','mlp_fused.csl':'performance/csl/mlp_fused.csl','fp8_encode.csl':'performance/csl/fp8_encode.csl'}.items():
 files[name]=(ROOT/path).read_bytes();origins[name]=dict(path=path,sha256=hashlib.sha256(files[name]).hexdigest())
manifest=dict(schema='wse-mlp-executable-candidate-v1',model=graph['model'],revision=graph['revision'],
              application=[750,1160],fabric=[762,1172],offset=[4,1],context=96,mlp_layer_argument_range=[0,63],
              parameter_residency='mlp-residency-plan-001.json',norm_slots_verified=True,
              code_roles=['paired original gate/up and down native worker','fused activation/max/scale/encoded gather','credited collector/header or up relay','preceding residual/post-attention RMS and residual/next RMS consumer','spectator original bank'],
              sources=origins,source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
              generator_sha256=hashlib.sha256((ROOT/'performance/spatial/mlp_layout.py').read_bytes()).hexdigest(),
              full_wafer_compiled=False,full_wafer_sram_admitted=False,executed=False,physical=False,full_model=False,
              missing=['full layout compiler admission','original bank loader/fixture and independent ordered full MLP oracle','complete numerical execution and replay','same-controller timing','production all-layer controller and termination','other operator programs/routes and correct dependent sentence generation'])
a.output.mkdir()
for n,b in files.items():(a.output/n).write_bytes(b)
(a.output/'candidate.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(layout_bytes=len(files['layout.csl']),sources=len(files),application=[750,1160],full_wafer_compiled=False,executed=False)))
