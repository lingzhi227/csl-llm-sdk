"""Pinned ordinary RMS/partial RoPE bodies with unchanged arithmetic observations."""
import ast,copy,hashlib,json,sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace,MethodType
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.qk_rope_numerics import POLICY,fixtures,reference,check_interval,bits
from qwen38.rotary_trig import grid_check,analytic_budget,HI,LO,SIN,COS
root=Path.cwd();torch.set_num_threads(1);torch.set_num_interop_threads(1)
raw=(root/'modeling_qwen3_5.py').read_bytes()
if hashlib.sha256(raw).hexdigest()!='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b':raise ValueError('Pinned source mismatch')
tree=ast.parse(raw)
rotary=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5TextRotaryEmbedding')
rms=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5RMSNorm')
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('rotate_half','apply_rotary_pos_emb')]
methods=[n for n in rotary.body if isinstance(n,ast.FunctionDef) and n.name in ('compute_default_rope_parameters','forward','recomposition_frequencies')]
trace={}
def observe(label,value):
    if isinstance(value,torch.Tensor):trace[label]=value.detach().clone()
    return value
class Hooks(ast.NodeTransformer):
    def visit_ClassDef(self,node):
        node.decorator_list=[]
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        node.decorator_list=[]
        return self.generic_visit(node)
    def visit_BinOp(self,node):
        label=f'{node.lineno}:{node.col_offset}:{type(node.op).__name__}'
        node=self.generic_visit(node)
        return ast.copy_location(ast.Call(func=ast.Name(id='observe',ctx=ast.Load()),args=[ast.Constant(label),node],keywords=[]),node)
selected=[rms,*functions,*methods]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*[Hooks().visit(copy.deepcopy(n)) for n in selected]],type_ignores=[])
ns=dict(torch=torch,nn=nn,observe=observe,maybe_autocast=lambda **kwargs:nullcontext())
exec(compile(ast.fix_missing_locations(module),'pinned_qk_rope_observed','exec'),ns)
config=SimpleNamespace(rope_parameters={'rope_theta':10000000,'partial_rotary_factor':0.25},head_dim=256)
frequencies,scaling=ns['compute_default_rope_parameters'](config)
if frequencies.dtype!=torch.float32 or frequencies.shape!=(32,) or scaling!=1.:raise ValueError('Static frequency contract')
rope=SimpleNamespace(inv_freq=frequencies,attention_scaling=scaling,mrope_section=[11,11,10])
rope.recomposition_frequencies=MethodType(ns['recomposition_frequencies'],rope)
data=fixtures();data['inverse_frequencies']=frequencies.tolist();data['inverse_frequency_bits']=[bits(x) for x in frequencies.tolist()]
trig=grid_check()
if not trig['passed']:raise ValueError('Source trig gate failed before SDK')
rows=[];oracles=[]
apply=next(n for n in functions if n.name=='apply_rotary_pos_emb')
product_labels={}
for name in ('q_embed','k_embed'):
    expr=next(n.value for n in apply.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    product_labels[name]=[f'{n.lineno}:{n.col_offset}:{type(n.op).__name__}' for n in (expr.left,expr.right,expr)]
for call,t in enumerate(data['tokens'],1):
    ref=reference(t,data['q_weight'],data['k_weight'],data['inverse_frequencies']);normalized=[];rms_dtype=[]
    for head in ('q','k'):
        layer=ns['Qwen3_5RMSNorm'](256,eps=POLICY['epsilon_f32'])
        layer.weight=nn.Parameter(torch.tensor(data[head+'_weight'],dtype=torch.bfloat16),requires_grad=False)
        x=torch.tensor(t[head],dtype=torch.bfloat16).reshape(1,1,256)
        trace.clear();y=layer(x);normalized.append(y)
        rms_dtype.append({k:str(v.dtype) for k,v in trace.items()})
        if y.dtype!=torch.bfloat16 or any(v.dtype!=torch.float32 for v in trace.values()):raise ValueError('RMS intermediate dtype contract')
    pos=torch.full((3,1,1),t['position'],dtype=torch.int64)
    cosine,sine=ns['forward'](rope,normalized[0],pos)
    if cosine.dtype!=torch.bfloat16 or sine.dtype!=torch.bfloat16:raise ValueError('BF16 trig boundary')
    trig_checks={'cosine':check_interval(cosine.float().flatten().tolist(),ref['cosine_bf16']*2,ref['cosine_bounds']*2),
                 'sine':check_interval(sine.float().flatten().tolist(),ref['sine_bf16']*2,ref['sine_bounds']*2)}
    trace.clear();outputs=ns['apply_rotary_pos_emb'](normalized[0].unsqueeze(1),normalized[1].unsqueeze(1),cosine,sine)
    product_dtypes={name:[str(trace[label].dtype) for label in labels] for name,labels in product_labels.items()}
    if any(dtype!='torch.bfloat16' for values in product_dtypes.values() for dtype in values):raise ValueError('Separate product/add BF16 boundary')
    observed=[];checks=[]
    for index,out in enumerate(outputs):
        actual=out.float().flatten().tolist();nr=normalized[index].float().flatten().tolist();h=ref['heads'][index]
        comparisons=dict(rms=check_interval(nr,h['rms']['output'],h['rms']['bounds']),output=check_interval(actual,h['output'],h['bounds']))
        tail=[bits(v) for v in actual[64:]]==[bits(v) for v in nr[64:]]
        position_zero_identity=t['position']!=0 or actual==nr  # Signed-zero behavior still checked by individual source casts on device.
        checks.append(dict(head=index,comparisons=comparisons,tail_exact=tail,position_zero_identity=position_zero_identity,
          nominal_bf16_mismatches=sum(bits(a)!=bits(b) for a,b in zip(actual,h['output'])),
          interval={k:h['interval'][k] for k in ('max_count','single_value_elements','max_numeric_span')}))
        observed.append(dict(rms=nr,output=actual,products=[trace[label].float().flatten().tolist() for label in product_labels[('q_embed','k_embed')[index]]]))
    if not all(c['passed'] for c in trig_checks.values()) or not all(all(c['passed'] for c in h['comparisons'].values()) and h['tail_exact'] and h['position_zero_identity'] for h in checks):raise ValueError('Official/source interval agreement')
    rows.append(dict(call=call,generation=t['generation'],position=t['position'],heads=checks,trig=trig_checks,rms_dtypes=rms_dtype,rotary_dtypes=product_dtypes))
    oracles.append(dict(source=ref,official=observed,official_cosine=cosine.float().flatten().tolist(),official_sine=sine.float().flatten().tolist()))
for name,value in (('fixtures.json',data),('oracles.json',oracles),('numerical-policy.json',POLICY),('trig-qualification.json',trig)):
    (root/name).write_text(json.dumps(value)+'\n')
lines=raw.decode().splitlines(True)
result=dict(passed=True,tokens=10,checks=rows,trig=trig,source_sha256=hashlib.sha256(raw).hexdigest(),
 upstream_commit='4815a0a6a064214f2d8208c094464a5a6b76ca8d',device='cpu',torch=torch.__version__,
 source_bodies={n.name:dict(span=[n.lineno,n.end_lineno],sha256=hashlib.sha256(''.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest()) for n in selected},
 adaptation='observation wrappers return unchanged values; integration decorators removed; fixed default rotary methods/equal text axes; nullcontext for disabled CPU autocast; no model/projection instantiation',
 inverse_frequency_bits=data['inverse_frequency_bits'],trig_coefficient_bits=dict(high=bits(HI),low=bits(LO),sin=[bits(float(x)) for x in SIN],cos=[bits(float(x)) for x in COS]),
 output_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('fixtures.json','oracles.json','numerical-policy.json','trig-qualification.json')})
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS10QKROPEREFERENCES',flush=True)
