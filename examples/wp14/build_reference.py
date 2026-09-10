"""Frozen original-hidden Q/K prefix and trained RMS/RoPE CPU observations."""
import ast
import copy
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType

import numpy as np
import torch
from torch import nn

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'core'))
from qwen38.trained_qk_rope_numerics import POLICY, reference, report, spans, fp32
from qwen38.rms_numerics import bits

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
plan = json.loads((ROOT/'reference-plan.json').read_text())
assert plan['policy'] == POLICY
for name, identity in plan['frozen_inputs'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == identity
raw = (ROOT/'modeling_qwen3_5.py').read_bytes()
tree = ast.parse(raw)

trace = {}
phase = ''


def observe(label, value):
    if isinstance(value, torch.Tensor):
        trace[phase+':'+label] = value.detach().clone()
    return value


class Hooks(ast.NodeTransformer):
    def visit_ClassDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        return self.generic_visit(node)

    def wrap(self, node, label):
        return ast.copy_location(ast.Call(func=ast.Name(id='observe', ctx=ast.Load()),
                                         args=[ast.Constant(label), node], keywords=[]), node)

    def visit_BinOp(self, node):
        label = f'{node.lineno}:{node.col_offset}:{type(node.op).__name__}'
        return self.wrap(self.generic_visit(node), label)

    def visit_Call(self, node):
        label = node.func.attr if isinstance(node.func, ast.Attribute) else ''
        result = self.generic_visit(node)
        if label in ('pow', 'mean', 'rsqrt'):
            result = self.wrap(result, f'{node.lineno}:{node.col_offset}:{label}')
        return result


rms = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5RMSNorm')
rotary = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5TextRotaryEmbedding')
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('rotate_half', 'apply_rotary_pos_emb')]
methods = [n for n in rotary.body if isinstance(n, ast.FunctionDef) and n.name in ('compute_default_rope_parameters', 'forward', 'recomposition_frequencies')]
selected = [rms, *functions, *methods]
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                         *[Hooks().visit(copy.deepcopy(n)) for n in selected]], type_ignores=[])
ns = dict(torch=torch, nn=nn, observe=observe, maybe_autocast=lambda **kwargs: nullcontext())
exec(compile(ast.fix_missing_locations(module), 'pinned_trained_qk_observed', 'exec'), ns)

attention = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5Attention')
forward = next(n for n in attention.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
prefix = []
for statement in forward.body:
    prefix.append(copy.deepcopy(statement))
    if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'key_states' for t in statement.targets):
        break
assert prefix[-1].targets[0].id == 'key_states'
prefix_code = compile(ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[])), 'pinned_projection_prefix', 'exec')


def decode(path, shape, offset=0):
    data = np.memmap(path, dtype='<u2', mode='r', offset=offset, shape=shape)
    return (np.asarray(data, dtype=np.uint32) << 16).view(np.float32)


weights_dir = Path(plan['weights_directory'])
weights = {name: decode(weights_dir/file, (rows, 5120), 512)
           for name, file, rows in [('q_proj', 'q-with-norm.bf16', 512), ('k_proj', 'k-with-norm.bf16', 256)]}
norm_weights = [decode(weights_dir/file, (256,)) for file in ('q-with-norm.bf16', 'k-with-norm.bf16')]
hidden = decode(ROOT/'hidden.bf16', (4, 5120))
projection = np.load(ROOT/'projection-observations.npz', allow_pickle=False)
model = SimpleNamespace(head_dim=256)
captured = {}


def before_norm(name):
    def hook(module, args):
        global phase
        phase = name
    return hook


def capture(name):
    def hook(module, args, output):
        captured[name] = output.detach().clone()
    return hook


for name in ('q_proj', 'k_proj'):
    layer = nn.Linear(5120, weights[name].shape[0], bias=False, dtype=torch.bfloat16)
    layer.weight = nn.Parameter(torch.from_numpy(weights[name]).to(torch.bfloat16), requires_grad=False)
    layer.register_forward_hook(capture(name))
    setattr(model, name, layer)
for name, weight in zip(('q_norm', 'k_norm'), norm_weights):
    layer = ns['Qwen3_5RMSNorm'](256, eps=POLICY['epsilon_f32'])
    layer.weight = nn.Parameter(torch.from_numpy(weight).to(torch.bfloat16), requires_grad=False)
    layer.register_forward_pre_hook(before_norm(name))
    layer.register_forward_hook(capture(name))
    setattr(model, name, layer)

config = SimpleNamespace(rope_parameters={'rope_theta': 10000000, 'partial_rotary_factor': 0.25}, head_dim=256)
frequencies, scaling = ns['compute_default_rope_parameters'](config)
assert frequencies.dtype == torch.float32 and frequencies.shape == (32,) and scaling == 1
rope = SimpleNamespace(inv_freq=frequencies, attention_scaling=scaling, mrope_section=[11, 11, 10])
rope.recomposition_frequencies = MethodType(ns['recomposition_frequencies'], rope)
apply = next(n for n in functions if n.name == 'apply_rotary_pos_emb')
product_labels = {}
for name in ('q_embed', 'k_embed'):
    expression = next(n.value for n in apply.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.BinOp)
                      and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    product_labels[name] = [f'rotary:{n.lineno}:{n.col_offset}:{type(n.op).__name__}'
                            for n in (expression.left, expression.right, expression)]

norm_body = next(n for n in rms.body if isinstance(n, ast.FunctionDef) and n.name == '_norm')
norm_expression = next(n.value for n in norm_body.body if isinstance(n, ast.Return))
rms_forward = next(n for n in rms.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
gain_expression = next(n.value for n in rms_forward.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.BinOp))
rms_nodes = dict(normalized=norm_expression, denominator=norm_expression.right.args[0],
                 inverse=norm_expression.right, mean=norm_expression.right.args[0].left,
                 squares=norm_expression.right.args[0].left.func.value,
                 gains=gain_expression.right, gained=gain_expression)
rms_labels = {name: f'{node.lineno}:{node.col_offset}:'+(type(node.op).__name__ if isinstance(node, ast.BinOp) else node.func.attr)
              for name, node in rms_nodes.items()}


def floats(tensor):
    return tensor.detach().float().numpy().astype(np.float64).reshape(-1)


def bounds_array(values):
    return np.array([(b.lo, b.hi) for b in values], dtype=np.float64)


arrays = {'norm_weights': np.concatenate(norm_weights), 'inverse_frequencies': frequencies.numpy()}
rows = []
for index, (case, position) in enumerate(zip(plan['cases'], plan['positions'])):
    selection = np.r_[0:256, 512:768]
    centers = projection[case+'__nominal'][selection].tolist()
    radii = projection[case+'__radius'][selection].tolist()
    # Construct all source bounds before observing the CPU prefix in this case.
    source = reference(centers, radii, np.concatenate(norm_weights).tolist(), frequencies.tolist(), position)
    trace.clear(); captured.clear()
    with torch.no_grad():
        xt = torch.from_numpy(hidden[index].copy()).to(torch.bfloat16).reshape(1, 1, 5120)
        env = dict(ns, self=model, hidden_states=xt)
        exec(prefix_code, env)
        phase = 'frequencies'
        cosine, sine = ns['forward'](rope, env['query_states'], torch.full((3, 1, 1), position, dtype=torch.int64))
        phase = 'rotary'
        outputs = ns['apply_rotary_pos_emb'](env['query_states'], env['key_states'], cosine, sine)
    observed_projection = np.concatenate((floats(captured['q_proj'])[:256], floats(captured['k_proj'])))
    assert np.array_equal(floats(env['gate']), floats(captured['q_proj'])[256:])
    assert np.array_equal(observed_projection, projection[case+'__official'][selection])
    checks = {'projection': report(observed_projection.tolist(), source['inputs'], centers),
              'sine': report(floats(sine)[:32].tolist(), source['sine'], source['sine_nominal']),
              'cosine': report(floats(cosine)[:32].tolist(), source['cosine'], source['cosine_nominal'])}
    rms_dtypes = {k: str(v.dtype) for k, v in trace.items() if k.startswith(('q_norm:', 'k_norm:'))}
    assert rms_dtypes and all(dtype == 'torch.float32' for dtype in rms_dtypes.values())
    product_dtypes = {name: [str(trace[label].dtype) for label in labels] for name, labels in product_labels.items()}
    assert all(dtype == 'torch.bfloat16' for values in product_dtypes.values() for dtype in values)
    assert cosine.dtype == sine.dtype == xt.dtype == torch.bfloat16
    assert all(captured[name].dtype == torch.bfloat16 for name in captured)
    case_arrays = dict(projection=observed_projection, projection_bounds=bounds_array(source['inputs']),
                       sine=floats(sine), cosine=floats(cosine), angle_bounds=bounds_array(source['angles']),
                       sine_bounds=bounds_array(source['sine']), cosine_bounds=bounds_array(source['cosine']))
    heads = []
    for h, (name, output) in enumerate(zip(('q', 'k'), outputs)):
        nr = floats(captured[name+'_norm']); actual = floats(output)
        ref = source['heads'][h]
        head_checks = {'normalized': report(nr.tolist(), ref['rms']['output'], ref['rms']['nominal']),
                       'output': report(actual.tolist(), ref['output'], ref['nominal'])}
        source_stage_bounds = dict(normalized=ref['rms']['normalized'], denominator=[ref['rms']['stats'][2]],
                                   inverse=[ref['rms']['stats'][4]], mean=[ref['rms']['stats'][1]],
                                   squares=[fp32(v.square()) for v in source['inputs'][h*256:(h+1)*256]],
                                   gains=ref['rms']['gains'], gained=ref['rms']['gained'])
        cx = centers[h*256:(h+1)*256]; cw = norm_weights[h].tolist()
        ideal_mean = math.fsum(v*v for v in cx)/256
        ideal_denominator = ideal_mean+POLICY['epsilon_f32']
        ideal_inverse = 1/math.sqrt(ideal_denominator)
        source_stage_nominals = dict(normalized=[v*ideal_inverse for v in cx], denominator=[ideal_denominator],
                                     inverse=[ideal_inverse], mean=[ideal_mean], squares=[v*v for v in cx],
                                     gains=[1+w for w in cw], gained=[v*ideal_inverse*(1+w) for v, w in zip(cx, cw)])
        for stage, label in rms_labels.items():
            values = floats(trace[name+'_norm:'+label]).tolist()
            head_checks['fp32_'+stage] = report(values, source_stage_bounds[stage])
            head_checks['fp32_'+stage]['max_nominal_difference'] = max(abs(a-b) for a, b in zip(values, source_stage_nominals[stage]))
        for p, label in enumerate(product_labels[name+'_embed']):
            # Pinned eager product/add each round to BF16. The source sum interval
            # is pre-BF16, so its observed BF16 value is checked by final output.
            values = floats(trace[label])
            case_arrays[name+f'_rotary_eager_{p}'] = values
            if p < 2:
                head_checks['product'+str(p)] = report(values.tolist(), ref['product'+str(p)])
        tail = np.array_equal(actual[64:].view(np.uint64), nr[64:].view(np.uint64))
        exact_zero = case != 'zero_after_nonzero' or bool(np.all(actual == 0))
        assert tail and exact_zero and (position != 0 or np.array_equal(actual, nr))
        counterexamples = {
            'zero': report([0.]*256, ref['output'])['outside'],
            'cyclic_shift_by_one': report(np.roll(actual, -1).tolist(), ref['output'])['outside'],
            'skip_normalization': report(observed_projection[h*256:(h+1)*256].tolist(), ref['output'])['outside'],
            'skip_rotation_first64': report(nr[:64].tolist(), ref['output'][:64])['outside'],
        }
        heads.append(dict(head=name, checks=head_checks, interval=spans(ref['output']),
                          normalized_interval=spans(ref['rms']['output']), counterexample_rejected_elements=counterexamples,
                          tail_exact=tail, structural_zero=exact_zero, projection_radius_max=max(radii[h*256:(h+1)*256])))
        for key, values in dict(normalized=nr, output=actual, nominal=np.array(ref['nominal']),
                                normalized_bounds=bounds_array(ref['rms']['output']),
                                output_bounds=bounds_array(ref['output']), stats_bounds=bounds_array(ref['rms']['stats']),
                                pre_gain_bounds=bounds_array(ref['rms']['normalized']), gained_bounds=bounds_array(ref['rms']['gained']),
                                product0_bounds=bounds_array(ref['product0']), product1_bounds=bounds_array(ref['product1']),
                                sum_bounds=bounds_array(ref['sums'])).items():
            case_arrays[name+'_'+key] = values
    for name, tensor in trace.items():
        case_arrays['trace_'+name] = floats(tensor)
    arrays.update({case+'__'+name: value for name, value in case_arrays.items()})
    passed = all(c['passed'] for c in checks.values()) and all(c['passed'] for h in heads for c in h['checks'].values())
    rows.append(dict(case=case, position=position, passed=passed, checks=checks, heads=heads,
                     rms_dtypes=rms_dtypes, rotary_dtypes=product_dtypes))

np.savez(ROOT/'observations.npz', **arrays)
lines = raw.decode().splitlines(True)
result = dict(passed=all(row['passed'] for row in rows), scope='CPU original-hidden trained Q/K projection through RMS and partial RoPE; no SDK execution',
              policy=POLICY, checks=rows, projection_oracle_sha256=plan['frozen_inputs']['projection-observations.npz'],
              source_sha256=hashlib.sha256(raw).hexdigest(), prefix_lines=[prefix[0].lineno, prefix[-1].end_lineno],
              source_bodies={n.name: dict(span=[n.lineno, n.end_lineno], sha256=hashlib.sha256(''.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest()) for n in selected},
              adaptation='Pinned selected projection/view/chunk/QK RMS prefix and rotary statements; observation wrappers return unchanged tensors; only integration decorators removed; equal text axes and disabled CPU autocast.',
              inverse_frequency_bits=[bits(f) for f in frequencies.tolist()],
              backend=dict(torch=torch.__version__, numpy=np.__version__, device='cpu', threads=torch.get_num_threads(), configuration=torch.__config__.show()),
              output_sha256=hashlib.sha256((ROOT/'observations.npz').read_bytes()).hexdigest(), output_bytes=(ROOT/'observations.npz').stat().st_size)
(ROOT/'result.json').write_text(json.dumps(result, indent=2)+'\n')
assert result['passed'], 'Original-hidden source interval failure; preserve observations and report before a changed candidate'
print('PASS four original-hidden Q/K preprocessing CPU references', flush=True)
