"""Unchanged official gated norm body, BF16 weight profile, independent math."""
import ast,hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.gated_numerics import POLICY,fixtures,reference,expanded
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=Path.cwd();raw=(root/'modeling_qwen3_5.py').read_bytes()
if hashlib.sha256(raw).hexdigest()!='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b':raise ValueError('Pinned source mismatch')
node=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5RMSNormGated')
node.decorator_list=[]  # Disable optional external hub dispatch; retain class/method bodies.
ns={'torch':torch,'nn':nn,'ACT2FN':{'silu':F.silu}}
tree=ast.Module(body=[node],type_ignores=[]);exec(compile(ast.fix_missing_locations(tree),'modeling_qwen3_5.py','exec'),ns)
cases=fixtures();rows=[]
for case in cases:
    x,w,z=[torch.tensor(case[n],dtype=torch.bfloat16).reshape(1,128) for n in ('x','w','z')]
    norm=ns[node.name](128).to(dtype=torch.bfloat16);norm.weight.data.copy_(w[0])
    with torch.no_grad():output=norm(x,z)
    ref=reference(case['x'],case['w'],case['z'])
    expected=torch.tensor([expanded(b) for b in ref['output_bits']],dtype=torch.float64).reshape(1,128)
    errors=(output.double()-expected).abs();ok=bool(torch.isfinite(output).all() and (errors<=2**-6*expected.abs()+1e-7).all())
    # Establish actual promotion behavior for the selected BF16 parameter profile.
    early=(x.float()*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+norm.variance_epsilon)).to(x.dtype)
    gain=norm.weight*early;pre=gain*F.silu(z.float())
    if gain.dtype!=torch.bfloat16 or pre.dtype!=torch.float32 or not torch.equal(pre.to(torch.bfloat16),output):raise ValueError('Dtype/source boundary mismatch')
    rows.append({'name':case['name'],'passed':ok,'official_dtype':str(output.dtype),'gain_product_dtype':str(gain.dtype),
       'final_precast_dtype':str(pre.dtype),'max_absolute_error':float(errors.max()),
       'oracle_bf16_mismatches_non_gating':int((output.double()!=expected).sum()),'adversaries':ref['adversaries'],
       'official_output':output.float().flatten().tolist()})
    if not ok:raise ValueError('Independent official-reference gate failed')
if not all(any(row['adversaries'][k]>0 for row in rows) for k in ('late_cast','offset_gain','sigmoid_only')):raise ValueError('Missing adversary')
(root/'fixtures.json').write_text(json.dumps(cases)+'\n');(root/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
result={'passed':True,'checks':rows,'source_sha256':hashlib.sha256(raw).hexdigest(),'source_lines':[node.lineno,node.end_lineno],
 'upstream_commit':'4815a0a6a064214f2d8208c094464a5a6b76ca8d','adaptation':'remove hub decorator only; unchanged class/method bodies; explicit torch/nn/ACT2FN; BF16 parameters; no full runtime',
 'torch':torch.__version__,'device':'cpu','output_hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('fixtures.json','numerical-policy.json')}}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows))
