"""Selected original head, full 5120 contraction; CPU evidence, no SDK execution."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path.cwd()/'core'))
from qwen38.interval_numerics import quantized_vector, check_interval, output_bf16_spans
from qwen38.attention_numerics import exact_sum_products

root = Path.cwd()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
spec = json.loads((root/'reference-plan.json').read_text())
weights_dir = Path(spec['weights_directory'])
receipt = json.loads((weights_dir/'download.json').read_text())
assert receipt['status'] == 'complete'
for name, identity in spec['weight_files'].items():
    assert hashlib.sha256((weights_dir/name).read_bytes()).hexdigest() == identity['sha256']
raw = (root/'modeling_qwen3_5.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == spec['source_sha256']
assert hashlib.sha256((root/'hidden.bf16').read_bytes()).hexdigest() == spec['hidden_sha256']

def decode(path, shape, offset=0):
    words = np.memmap(path, dtype='<u2', mode='r', offset=offset, shape=shape)
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)

weights = {name: decode(weights_dir/t['file'], (t['selected_rows'][1], 5120) if len(t['shape']) == 2 else (256,), t['file_offset'])
           for name, t in receipt['plan']['tensors'].items()}
hidden = decode(root/'hidden.bf16', (4, 5120))
matrix = np.concatenate([weights[n] for n in ('q_proj', 'k_proj', 'v_proj')]).astype(np.float64)
assert matrix.shape == (1024, 5120)
assert np.isfinite(matrix).all()

# Execute the pinned source's projection/view/chunk/RMS prefix. Only the selected
# rows are instantiated. No full model, cache, output projection or GPU is involved.
tree = ast.parse(raw)
rms = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5RMSNorm'))
rms.decorator_list = []
attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5Attention')
forward = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
prefix = []
for statement in forward.body:
    prefix.append(copy.deepcopy(statement))
    if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'value_states' for t in statement.targets):
        break
ns = {'torch': torch, 'nn': nn}
exec(compile(ast.fix_missing_locations(ast.Module(body=[rms], type_ignores=[])), 'pinned_rms', 'exec'), ns)
prefix_code = compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), 'pinned_projection_prefix', 'exec')
module = SimpleNamespace(head_dim=256)
for name in ('q_proj', 'k_proj', 'v_proj'):
    layer = nn.Linear(5120, weights[name].shape[0], bias=False, dtype=torch.bfloat16)
    layer.weight = nn.Parameter(torch.from_numpy(weights[name]).to(torch.bfloat16), requires_grad=False)
    setattr(module, name, layer)
for name in ('q_norm', 'k_norm'):
    layer = ns['Qwen3_5RMSNorm'](256, eps=1e-6)
    layer.weight = nn.Parameter(torch.from_numpy(weights[name]).to(torch.bfloat16), requires_grad=False)
    setattr(module, name, layer)

arrays = {}
rows = []
u = 2**-24
gamma = 5120*u/(1-5120*u)
double_gamma = 5120*2**-53/(1-5120*2**-53)
for case, x in zip(spec['cases'], hidden):
    xd = x.astype(np.float64)
    # Independent row-major FP64 products/sums; temporary is only 40 MiB.
    products = matrix*xd
    center = np.sum(products, axis=1, dtype=np.float64)
    magnitude = np.sum(np.abs(products), axis=1, dtype=np.float64)
    reference_error = double_gamma*magnitude
    error = (gamma*magnitude/(1-double_gamma)+reference_error+5120*2**-149)*(1+2**-40)
    exact = case in ('last_column_onehot', 'zero_after_nonzero')
    if exact:
        expected = matrix[:, 5119] if case == 'last_column_onehot' else np.zeros(1024)
        assert np.array_equal(center, expected)
        error.fill(0)
    nominal, radius = quantized_vector(center.tolist(), error.tolist())
    with torch.no_grad():
        xt = torch.from_numpy(x.copy()).to(torch.bfloat16).reshape(1, 1, 5120)
        projected = torch.cat([getattr(module, name)(xt).flatten() for name in ('q_proj', 'k_proj', 'v_proj')])
        env = dict(ns, self=module, hidden_states=xt)
        exec(prefix_code, env)
    observed = projected.float().numpy().astype(np.float64)
    comparison = check_interval(observed.tolist(), nominal, radius)
    assert comparison['passed'], comparison
    assert np.array_equal(env['gate'].float().numpy().flatten(), observed[256:512])
    assert np.array_equal(env['value_states'].float().numpy().flatten(), observed[768:])
    normalized = [env[name].float().numpy().flatten() for name in ('query_states', 'key_states')]
    domain = {
        'q_input_abs_le1': bool(np.max(np.abs(observed[:256])) <= 1),
        'k_input_abs_le1': bool(np.max(np.abs(observed[512:768])) <= 1),
        'v_abs_le1': bool(np.max(np.abs(observed[768:])) <= 1),
        'gate_abs_le8': bool(np.max(np.abs(observed[256:512])) <= 8),
        'q_norm_offset_in_minus2_to_half': bool(np.all((weights['q_norm'] >= -2) & (weights['q_norm'] <= .5))),
        'k_norm_offset_in_minus2_to_half': bool(np.all((weights['k_norm'] >= -2) & (weights['k_norm'] <= .5))),
        'q_square_fp32_exact_proof': exact_sum_products(observed[:256].tolist(), observed[:256].tolist()),
        'k_square_fp32_exact_proof': exact_sum_products(observed[512:768].tolist(), observed[512:768].tolist()),
        'nominal_normalized_max_abs': [float(np.max(np.abs(a))) for a in normalized],
        'composition_qualified': False,
    }
    span = output_bf16_spans(nominal, radius)
    rows.append(dict(case=case, comparison=comparison, exact_structural=exact,
                     nominal_bf16_mismatches=int(np.count_nonzero(observed != nominal)),
                     max_abs_product_sum=float(magnitude.max()), max_precast_error=float(error.max()),
                     interval={k:span[k] for k in ('max_count','single_value_elements','max_numeric_span')},
                     domain=domain, projection_dtype=str(projected.dtype), hidden_dtype=str(xt.dtype)))
    for name, values in dict(fp64=center, absolute_products=magnitude, precast_error=error,
                             nominal=np.array(nominal), radius=np.array(radius), official=observed,
                             normalized=np.array(normalized)).items():
        arrays[case+'__'+name] = values

np.savez(root/'observations.npz', **arrays)
result = dict(passed=True, scope='original selected head projections with synthetic full5120 hidden inputs',
              cases=rows, policy=spec['arithmetic'], source_sha256=spec['source_sha256'],
              prefix_lines=[prefix[0].lineno, prefix[-1].end_lineno],
              adaptation='Pinned projection/view/split/RMS statements unchanged; selected row modules only; RMS integration decorator removed.',
              backend={'torch':torch.__version__, 'numpy':np.__version__, 'device':'cpu',
                       'threads':torch.get_num_threads(), 'configuration':torch.__config__.show(),
                       'dispatch_caveat':'CPU BF16 Linear backend accumulation tree is not asserted; observed outputs checked against serial-FMA source enclosure.'},
              weight_domains={name: {'min':float(a.min()), 'max':float(a.max())} for name,a in weights.items()},
              composition_blockers=['Original K norm offset exceeds accepted upper bound0.5.',
                  'Accepted source RMS requires exact-square fixtures; original projected rows require a new full error policy.',
                  'Projection uncertainty in Q/K/V/gate must propagate through normalization, rotary, cache and attention; nominal checks do not qualify composition.'],
              observations_sha256=hashlib.sha256((root/'observations.npz').read_bytes()).hexdigest())
(root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print('PASS selected original projection references; composition remains unqualified', flush=True)
