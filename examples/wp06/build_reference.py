"""Freeze synthetic effective inputs and a matching extracted upstream recurrence."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.recurrent_numerics import POLICY,initial_state,raw_tokens,step
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=Path.cwd();raw=(root/'modeling_qwen3_5.py').read_bytes()
if hashlib.sha256(raw).hexdigest()!='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b':raise ValueError('Source mismatch')
nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('l2norm','torch_recurrent_gated_delta_rule')]
spans={n.name:[n.lineno,n.end_lineno] for n in nodes}
for n in nodes:n.decorator_list=[]
tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
namespace={'torch':torch};exec(compile(ast.fix_missing_locations(tree),'modeling_qwen3_5.py','exec'),namespace)
fn=namespace['torch_recurrent_gated_delta_rule'];initial=initial_state();official_state=None
previous=initial;previous_bounds=[0.0]*(128*128);last_generation=0;tokens=[];oracles=[];checks=[]
for token in raw_tokens():
    generation=token['generation']
    if generation!=last_generation:
        previous=initial if generation==1 else [0.0]*(128*128)
        previous_bounds=[0.0]*(128*128)
        official_state=torch.tensor(previous,dtype=torch.float32).reshape(1,1,128,128)
        last_generation=generation
    query=torch.tensor(token['raw_q'],dtype=torch.float32).reshape(1,1,1,128)
    key=torch.tensor(token['k'],dtype=torch.float32).reshape(1,1,1,128)
    value=torch.tensor(token['v'],dtype=torch.float32).reshape(1,1,1,128)
    g=torch.tensor([[[math.log(token['requested_decay'])]]],dtype=torch.float32)
    beta=torch.tensor([[[token['beta']]]],dtype=torch.float32)
    # These exact float32 operands match operations inside the unchanged official body.
    effective_q=query/(128**0.5);effective_decay=g.exp()
    token.update(q=effective_q.flatten().tolist(),decay=float(effective_decay.item()),official_log_decay=float(g.item()),
                 requested_decay_minus_effective=token['requested_decay']-float(effective_decay.item()),
                 raw_q_norm=math.sqrt(math.fsum(x*x for x in token['raw_q'])),k_norm=math.sqrt(math.fsum(x*x for x in token['k'])))
    reference=step(previous,previous_bounds,token)
    output,official_state=fn(query,key,value,g,beta,initial_state=official_state,output_final_state=True,use_qk_l2norm_in_kernel=False)
    for name,actual in (('output',output.flatten()),('state',official_state.flatten())):
        expected=torch.tensor(reference[name],dtype=torch.float64);error=(actual.double()-expected).abs()
        ok=bool(torch.isfinite(actual).all() and (error<=3e-6+3e-6*expected.abs()).all())
        checks.append({'generation':generation,'token':token['token'],'stage':name,'passed':ok,'max_absolute_error':float(error.max())})
        if not ok:raise ValueError('Official body mismatch')
    reference['official_output']=output.flatten().tolist();reference['official_state']=official_state.flatten().tolist()
    tokens.append(token);oracles.append(reference);previous=reference['state'];previous_bounds=reference['state_bounds']
fixture={'initial_state':initial,'tokens':tokens,'scope':'synthetic normalized/scaled effective q, normalized k, explicit decay/beta/v'}
(root/'fixtures.json').write_text(json.dumps(fixture)+'\n');(root/'oracles.json').write_text(json.dumps(oracles)+'\n')
(root/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
result={'passed':all(x['passed'] for x in checks),'checks':checks,'torch':torch.__version__,'device':'cpu','threads':1,
        'source_sha256':hashlib.sha256(raw).hexdigest(),'upstream_commit':'4815a0a6a064214f2d8208c094464a5a6b76ca8d',
        'definition_lines':spans,'adaptation':'remove hub decorators only; unchanged official bodies; explicit torch dependency; no complete Transformers runtime',
        'boundary':'official normalization disabled; exact q division and exp(g) results become candidate FP32 inputs',
        'output_hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('fixtures.json','oracles.json','numerical-policy.json')}}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'checks':len(checks)}))
