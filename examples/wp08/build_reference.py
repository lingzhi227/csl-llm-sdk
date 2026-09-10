"""Pinned causal convolution/L2 functions and exact gate assignment AST nodes."""
import ast,hashlib,json,math,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.preprocess_numerics import POLICY,fixtures,history_step,reference
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=Path.cwd();raw=(root/'modeling_qwen3_5.py').read_bytes()
if hashlib.sha256(raw).hexdigest()!='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b':raise ValueError('Pinned source mismatch')
tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('causal_conv1d_update','causal_conv1d_fn','l2norm')]
spans={n.name:[n.lineno,n.end_lineno] for n in nodes}
for n in nodes:n.decorator_list=[]
ns={'torch':torch,'nn':nn,'F':F,'ACT2FN':{'silu':F.silu}}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'modeling_qwen3_5.py','exec'),ns)
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5GatedDeltaNet')
forward=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
gate_nodes=[n for n in forward.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('beta','g') for t in n.targets)]
if len(gate_nodes)!=2:raise ValueError('Expected exact beta/g assignments')
gate_code=compile(ast.fix_missing_locations(ast.Module(body=gate_nodes,type_ignores=[])),'modeling_qwen3_5.py','exec')
data=fixtures();weight=torch.tensor(data['weights'],dtype=torch.bfloat16).reshape(384,4)
state=torch.zeros((1,384,4),dtype=torch.bfloat16);history=[0.0]*1536;generation=0;rows=[];oracles=[];sequence=[]
for t in data['tokens']:
    if generation!=t['generation']:
        state.zero_();history=[0.0]*1536;generation=t['generation'];sequence=[]
    hidden=torch.tensor(t['x'],dtype=torch.bfloat16).reshape(1,384,1)
    sequence.append(hidden)
    output=ns['causal_conv1d_update'](hidden,state,weight,None,'silu')
    history=history_step(history,t['x']);ref=reference(history,data['weights'],t)
    if state.float().flatten().tolist()!=history:raise ValueError('Official history mismatch')
    # A separate full causal replay must agree with cached update on every prefix.
    whole=ns['causal_conv1d_fn'](torch.cat(sequence,dim=-1),weight,None,'silu')
    if not torch.equal(whole[:,:,-1:],output):raise ValueError('Cached/full causal orientation mismatch')
    values=output.float().flatten();q=ns['l2norm'](values[:128].reshape(1,128))/math.sqrt(128)
    k=ns['l2norm'](values[128:256].reshape(1,128));v=values[256:]
    gate_env={'self':SimpleNamespace(A_log=torch.tensor([t['A_log']],dtype=torch.bfloat16),dt_bias=torch.tensor([t['dt_bias']],dtype=torch.bfloat16)),
      'a':torch.tensor([t['a']],dtype=torch.bfloat16),'b':torch.tensor([t['b']],dtype=torch.bfloat16),'F':F}
    exec(gate_code,gate_env);beta=gate_env['beta'];g=gate_env['g'];decay=g.exp()
    if beta.dtype!=torch.bfloat16 or g.dtype!=torch.float32:raise ValueError('Gate promotion mismatch')
    actual={'q':q.flatten().tolist(),'k':k.flatten().tolist(),'v':v.tolist(),'beta':[float(beta.item())],'g':[float(g.item())],'decay':[float(decay.item())]}
    checks={}
    for name,a in actual.items():
        expected=ref[name] if isinstance(ref[name],list) else [ref[name]]
        errors=[abs(x-y) for x,y in zip(a,expected)]
        budget=2**-6 if name in ('q','k','v','beta') else 2**-18
        atol=2e-6 if name in ('q','k','v','beta') else 1e-10
        checks[name]={'passed':all(math.isfinite(x) and e<=atol+budget*abs(y) for x,y,e in zip(a,expected,errors)),
          'max_absolute_error':max(errors),'relative_budget':budget,'absolute_budget':atol}
    if not all(x['passed'] for x in checks.values()):raise ValueError('Independent preprocessing comparison failed')
    rows.append({'generation':generation,'token':t['token'],'checks':checks,'history_exact':True,'cached_full_prefix_exact':True,
       'beta_dtype':str(beta.dtype),'g_dtype':str(g.dtype),'actual':actual})
    oracles.append({'history':history,'reference':ref})
(root/'fixtures.json').write_text(json.dumps(data)+'\n');(root/'oracles.json').write_text(json.dumps(oracles)+'\n')
(root/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
result={'passed':True,'tokens':8,'checks':rows,'source_sha256':hashlib.sha256(raw).hexdigest(),
 'upstream_commit':'4815a0a6a064214f2d8208c094464a5a6b76ca8d','function_lines':spans,'gate_assignment_lines':[[n.lineno,n.end_lineno] for n in gate_nodes],
 'adaptation':'remove optional function hub decorators only; unchanged function and gate-assignment bodies; explicit dependencies; no complete model runtime',
 'torch':torch.__version__,'device':'cpu','output_hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('fixtures.json','oracles.json','numerical-policy.json')}}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS8TOKENS')
