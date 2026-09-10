"""Pinned eager attention observations and independent original-input intervals."""
import ast,copy,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.attention_numerics import POLICY,fixtures,append,reference,check_interval,bits,bf16_of_real
root=Path.cwd();torch.set_num_threads(1);torch.set_num_interop_threads(1)
raw=(root/'modeling_qwen3_5.py').read_bytes()
if hashlib.sha256(raw).hexdigest()!='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b':raise ValueError('Pinned source mismatch')
tree=ast.parse(raw);original=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('repeat_kv','eager_attention_forward')]
if len(original)!=2:raise ValueError('Missing source functions')
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5Attention')
forward=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
gate_assignment=next(n for n in forward.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp) and isinstance(n.value.right,ast.Call) and isinstance(n.value.right.func,ast.Attribute) and n.value.right.func.attr=='sigmoid')
trace={}
def observe(name,value):trace[name]=value.detach().clone();return value
class ObservationHooks(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node);label=None
        if isinstance(node.func,ast.Attribute):
            if node.func.attr=='matmul':label='qk_matmul' if isinstance(node.args[0],ast.Name) and node.args[0].id=='query' else 'value_matmul'
            elif node.func.attr=='softmax':label='softmax'
            elif node.func.attr=='sigmoid':label='output_gate'
        return ast.copy_location(ast.Call(func=ast.Name(id='observe',ctx=ast.Load()),args=[ast.Constant(label),node],keywords=[]),node) if label else node
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Mult) and isinstance(node.value.right,ast.Name) and node.value.right.id=='scaling':
            return [node,ast.Expr(value=ast.Call(func=ast.Name(id='observe',ctx=ast.Load()),args=[ast.Constant('scaled_scores'),ast.Name(id='attn_weights',ctx=ast.Load())],keywords=[]))]
        return node
nodes=[ObservationHooks().visit(copy.deepcopy(n)) for n in original]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
ns=dict(torch=torch,nn=nn,observe=observe)
exec(compile(ast.fix_missing_locations(module),'pinned_eager_observed','exec'),ns)
gate_node=ObservationHooks().visit(copy.deepcopy(gate_assignment))
gate_code=compile(ast.fix_missing_locations(ast.Module(body=[gate_node],type_ignores=[])),'pinned_gate_observed','exec')
data=fixtures();cache=data['initial_cache'].copy();generation=0;q_prefix=[];rows=[];oracles=[]
module=SimpleNamespace(num_key_value_groups=1,training=False)
required={'qk_matmul':'torch.bfloat16','scaled_scores':'torch.bfloat16','softmax':'torch.float32','value_matmul':'torch.bfloat16','output_gate':'torch.bfloat16','gated_output':'torch.bfloat16'}
for call,t in enumerate(data['tokens'],1):
    if generation!=t['generation']:generation=t['generation'];q_prefix=[]
    cache=append(cache,t);q_prefix.append(t['q']);n=t['token'];ref=reference(t,cache)
    q=torch.tensor(t['q'],dtype=torch.bfloat16).reshape(1,1,1,256)
    k=torch.tensor(cache[:n*256],dtype=torch.bfloat16).reshape(1,1,n,256)
    v=torch.tensor(cache[2048:2048+n*256],dtype=torch.bfloat16).reshape(1,1,n,256)
    trace.clear();out,prob=ns['eager_attention_forward'](module,q,k,v,None,1/16)
    env={**ns,'attn_output':out.reshape(1,1,256),'gate':torch.tensor(t['gate'],dtype=torch.bfloat16).reshape(1,1,256)}
    exec(gate_code,env);observe('gated_output',env['attn_output'])
    observed_dtypes={name:str(value.dtype) for name,value in trace.items()}
    if observed_dtypes!=required:raise ValueError('Source dtype contract discrepancy: '+str(observed_dtypes))
    official=env['attn_output'].float().flatten().tolist();attention=out.float().flatten().tolist();probability=prob.float().flatten().tolist()
    source_checks={'output':check_interval(official,ref['output'],ref['output_bounds']),
      'attention':check_interval(attention,ref['attention_bf16'],ref['attention_bounds']),
      'probability':check_interval(probability,ref['probability_bf16'],ref['probability_bounds'])}
    # An explicit full causal matrix, evaluated independently on original prefix
    # queries, must agree with the same source interval for its current last row.
    qall=torch.tensor(q_prefix,dtype=torch.bfloat16).reshape(1,1,n,256)
    mask=torch.full((n,n),float('-inf'),dtype=torch.bfloat16).triu(1).reshape(1,1,n,n)
    causal,causal_prob=ns['eager_attention_forward'](module,qall,k,v,mask,1/16)
    causal_env={**ns,'attn_output':causal[:,-1:,:,:].reshape(1,1,256),'gate':env['gate']}
    exec(gate_code,causal_env);causal_output=causal_env['attn_output'].float().flatten().tolist()
    source_checks['explicit_causal_last_row']=check_interval(causal_output,ref['output'],ref['output_bounds'])
    if not all(c['passed'] for c in source_checks.values()):raise ValueError('Original-source interval agreement failed: '+str(source_checks))
    rows.append(dict(call=call,generation=generation,token=n,dtypes=observed_dtypes,checks=source_checks,
      exact_score_dots=sum(ref['score_exact']),valid_prefix=n,all_negative_scores=all(x<0 for x in ref['scaled_bf16']),
      nominal_bf16_mismatches=sum((bits(a)>>16)!=bf16_of_real(b) for a,b in zip(official,ref['output'])),
      causal_vs_decode_bf16_mismatches=sum(bits(a)!=bits(b) for a,b in zip(official,causal_output)),
      output_interval={k:ref['output_interval'][k] for k in ('max_count','single_value_elements','max_numeric_span')}))
    oracles.append(dict(cache=cache,source=ref,official_output=official,official_attention=attention,official_probabilities=probability))
if not rows[3]['all_negative_scores']:raise ValueError('Missing all-negative score fixture')
(root/'fixtures.json').write_text(json.dumps(data)+'\n');(root/'oracles.json').write_text(json.dumps(oracles)+'\n');(root/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
lines=raw.decode().splitlines(True)
result=dict(passed=True,tokens=10,source_sha256=hashlib.sha256(raw).hexdigest(),upstream_commit='4815a0a6a064214f2d8208c094464a5a6b76ca8d',
 definition_spans={n.name:[n.lineno,n.end_lineno] for n in original},gate_assignment_line=gate_assignment.lineno,
 adaptation='observation-only wrappers returning unchanged tensors; exact eager/gate arithmetic; future annotations; no model/projection instantiation',
 source_body_hashes={n.name:hashlib.sha256(''.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest() for n in original},
 checks=rows,torch=torch.__version__,device='cpu',output_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('fixtures.json','oracles.json','numerical-policy.json')})
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS10ATTENTIONREFERENCES',flush=True)
