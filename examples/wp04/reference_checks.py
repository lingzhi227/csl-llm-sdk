"""Bounded CPU fixtures against exact, selected upstream fallback bodies.

No model construction, weights, Transformers package import, hub kernels or GPU.
Decorators are removed from an AST copy; selected function/method bodies are intact.
"""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
root = Path.cwd()
source = root / 'modeling_qwen3_5.py'
raw = source.read_bytes()
if hashlib.sha256(raw).hexdigest() != '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b':
    raise ValueError('Pinned official source mismatch')
selected = {
    'Qwen3_5RMSNorm', 'Qwen3_5RMSNormGated', 'l2norm',
    'torch_recurrent_gated_delta_rule', 'torch_chunk_gated_delta_rule',
    'causal_conv1d_fn', 'causal_conv1d_update', 'rotate_half',
    'apply_rotary_pos_emb', 'repeat_kv', 'eager_attention_forward',
}
tree = ast.parse(raw)
nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in selected]
if {n.name for n in nodes} != selected:
    raise ValueError('Missing upstream definitions')
provenance = {n.name: {'start': n.lineno, 'end': n.end_lineno} for n in nodes}
for n in nodes:
    for child in ast.walk(n):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
            child.decorator_list = []
namespace = {'torch': torch, 'nn': nn, 'F': F, 'ACT2FN': {'silu': F.silu}}
helper_raw = (root / 'import_utils.py').read_bytes()
if hashlib.sha256(helper_raw).hexdigest() != 'e147f99cf68352e3260e9780930501c3becc0c40a5ff04150ff641b724161732':
    raise ValueError('Pinned upstream helper mismatch')
helper = next(n for n in ast.parse(helper_raw).body if isinstance(n, ast.FunctionDef) and n.name == 'is_torchdynamo_exporting')
provenance[helper.name] = {'file': 'import_utils.py', 'start': helper.lineno, 'end': helper.end_lineno}
exec(compile(ast.Module(body=[helper], type_ignores=[]), 'import_utils.py', 'exec'), namespace)
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
checks = []


def close(name, actual, expected, atol=2e-6, rtol=2e-6):
    actual, expected = actual.detach().double(), expected.detach().double()
    error = (actual - expected).abs()
    passed = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()
                  and (error <= atol + rtol * expected.abs()).all())
    checks.append({'name': name, 'passed': passed, 'elements': actual.numel(),
                   'max_absolute_error': float(error.max()), 'atol': atol, 'rtol': rtol})
    if not passed:
        raise AssertionError(name)


def truth(name, passed):
    checks.append({'name': name, 'passed': bool(passed)})
    if not passed:
        raise AssertionError(name)


with torch.no_grad():
    # Full hidden-size RMS, adversarial gain convention and BF16 output boundary.
    x = ((torch.arange(5120) % 29) - 14).float().reshape(1, -1) / 8
    w = ((torch.arange(5120) % 7) - 3).float() / 8
    norm = namespace['Qwen3_5RMSNorm'](5120)
    norm.weight.copy_(w)
    oracle = x.double() / torch.sqrt(x.double().square().mean(-1, keepdim=True) + 1e-6) * (1 + w.double())
    close('rms5120_offset_gain_fp32', norm(x), oracle)
    close('rms5120_bf16_output', norm.to(torch.bfloat16)(x.to(torch.bfloat16)), oracle.to(torch.bfloat16), 0, 0)
    close('rms_zero_input', norm(torch.zeros_like(x)), torch.zeros_like(x), 0, 0)
    norm.weight.zero_()
    truth('zero_weight_is_unit_gain_not_zero_output', bool(norm(x).abs().max() > 0))
    norm.weight.fill_(-1)
    close('minus_one_weight_annihilates', norm(x), torch.zeros_like(x), 0, 0)

    # DeltaNet gated RMS has direct gain, unlike the preceding zero-centered RMS.
    gx, gate = x[:, :128], torch.linspace(-2, 2, 128).reshape(1, -1)
    gated = namespace['Qwen3_5RMSNormGated'](128)
    gw = torch.linspace(-0.5, 1.5, 128)
    gated.weight.copy_(gw)
    gref = gx.double() / torch.sqrt(gx.double().square().mean(-1, keepdim=True) + 1e-6)
    gref = gref * gw.double() * gate.double() / (1 + torch.exp(-gate.double()))
    close('gated_rms128_direct_gain_silu', gated(gx, gate), gref)
    gated.weight.zero_()
    close('gated_zero_weight_zeroes_output', gated(gx, gate), torch.zeros_like(gx), 0, 0)
    z = torch.tensor([-2., 0., 2.])
    sigmoid, swish = torch.sigmoid(z), F.silu(z)
    truth('sigmoid_and_swish_same_raw_gate_not_equivalent', not torch.allclose(sigmoid, swish))
    gate_counterexample = {'raw_gate': z.tolist(), 'branch_input': [1., 1., 1.],
                           'sigmoid_branch': sigmoid.tolist(), 'swish_branch': swish.tolist()}

    # Full 128x128 recurrent head, two tokens, independent FP64 matrix update.
    q = ((torch.arange(256) % 17) - 8).reshape(1, 2, 1, 128).float() / 16
    k = ((torch.arange(256) % 19) - 9).reshape(1, 2, 1, 128).float() / 16
    value = ((torch.arange(256) % 13) - 6).reshape(1, 2, 1, 128).float() / 16
    decay = torch.tensor([[[-0.25], [-0.5]]])
    beta = torch.tensor([[[0.25], [0.75]]])
    state = ((torch.arange(128*128) % 23) - 11).reshape(1, 1, 128, 128).float() / 1024
    original = state.clone()
    fn = namespace['torch_recurrent_gated_delta_rule']
    actual, final = fn(q, k, value, decay, beta, state, True, True)
    oracle_state = state[0, 0].double().clone()
    ys = []
    for i in range(2):
        qi, ki, vi = q[0, i, 0].double(), k[0, i, 0].double(), value[0, i, 0].double()
        qi = qi / torch.sqrt(torch.dot(qi, qi) + 1e-6) / math.sqrt(128)
        ki = ki / torch.sqrt(torch.dot(ki, ki) + 1e-6)
        decayed = math.exp(float(decay[0, i, 0])) * oracle_state
        correction = float(beta[0, i, 0]) * (vi - ki @ decayed)
        oracle_state = decayed + torch.outer(ki, correction)
        ys.append(qi @ oracle_state)
    close('recurrent128_output_fp64_oracle', actual[0, :, 0], torch.stack(ys))
    close('recurrent128_state_fp64_oracle', final[0, 0], oracle_state)
    close('recurrent_initial_state_not_mutated', state, original, 0, 0)
    a1, s1 = fn(q[:, :1], k[:, :1], value[:, :1], decay[:, :1], beta[:, :1], state, True, True)
    a2, s2 = fn(q[:, 1:], k[:, 1:], value[:, 1:], decay[:, 1:], beta[:, 1:], s1, True, True)
    close('decode_two_steps_equals_sequence', torch.cat([a1, a2], 1), actual, 0, 0)
    close('decode_final_state_equals_sequence', s2, final, 0, 0)
    chunk, chunk_state = namespace['torch_chunk_gated_delta_rule'](
        q, k, value, decay, beta, initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    close('prefill_chunk_vs_decode_output', chunk, actual)
    close('prefill_chunk_vs_decode_state', chunk_state, final)
    zero, zero_state = fn(q*0, k*0, value*0, decay, beta, None, True, True)
    close('zero_input_zero_recurrent_state', zero_state, torch.zeros_like(final), 0, 0)
    close('zero_input_zero_recurrent_output', zero, torch.zeros_like(actual), 0, 0)

    # Causal convolution: left history, full kernel width, sequence vs cached steps.
    conv_x = torch.arange(18).reshape(1, 3, 6).float()/16
    conv_w = torch.tensor([[1., 2., 3., 4.], [-1., 0., 1., 2.], [0., 1., 0., -1.]])/8
    conv_full = namespace['causal_conv1d_fn'](conv_x, conv_w, activation='silu')
    history = torch.zeros(1, 3, 4)
    conv_steps = [namespace['causal_conv1d_update'](conv_x[:, :, i:i+1], history, conv_w, activation='silu') for i in range(6)]
    close('conv4_prefill_vs_cached_decode', torch.cat(conv_steps, -1), conv_full)
    close('conv4_final_history', history, conv_x[:, :, -4:], 0, 0)
    truth('conv4_first_position_has_no_future', bool(torch.allclose(conv_full[:, :, 0], F.silu(conv_x[:, :, 0]*conv_w[:, -1]))))

    # Full attention dimensions: 24 Q heads, 4 KV heads, 256 dimensions, two tokens.
    aq = ((torch.arange(24*2*256) % 17)-8).reshape(1, 24, 2, 256).float()/32
    ak = ((torch.arange(4*2*256) % 13)-6).reshape(1, 4, 2, 256).float()/32
    av = ((torch.arange(4*2*256) % 23)-11).reshape(1, 4, 2, 256).float()/32
    mask = torch.tensor([[[[0., float('-inf')], [0., 0.]]]])
    attn, probs = namespace['eager_attention_forward'](SimpleNamespace(num_key_value_groups=6, training=False), aq, ak, av, mask, 1/16)
    expected = torch.empty_like(attn, dtype=torch.float64)
    for h in range(24):
        for t in range(2):
            scores = torch.tensor([float(torch.dot(aq[0,h,t].double(), ak[0,h//6,j].double()))/16 for j in range(t+1)], dtype=torch.float64)
            exp = torch.exp(scores-scores.max());prob = exp/exp.sum()
            expected[0,t,h] = sum((prob[j]*av[0,h//6,j].double() for j in range(t+1)))
    close('attention24q4kv256_causal_fp64_oracle', attn, expected)
    close('future_probability_zero', probs[0,:,0,1], torch.zeros(24), 0, 0)
    close('first_token_uses_own_value', attn[0,0], av[0,:,0].repeat_interleave(6,dim=0), 0, 0)

    # Partial RoPE acts on 64 of 256 dimensions, split-half layout.
    inv = 1 / (10_000_000. ** (torch.arange(0,64,2).float()/64))
    angle = torch.arange(2).float().reshape(1,2,1)*inv
    cos, sin = torch.cat([angle.cos()]*2,-1), torch.cat([angle.sin()]*2,-1)
    rq, rk = namespace['apply_rotary_pos_emb'](aq, ak, cos, sin)
    close('rope_position_zero_identity', rq[:,:,0], aq[:,:,0], 0, 0)
    close('rope_unrotated_tail192', rq[:,:,:,64:], aq[:,:,:,64:], 0, 0)
    expected_pair = aq[0,0,1,0].double()*math.cos(1)-aq[0,0,1,32].double()*math.sin(1)
    close('rope_position_one_split_half_pair', rq[0,0,1,0], expected_pair)
    packed = torch.arange(24*512).reshape(1,1,24,512)
    pq, pg = torch.chunk(packed, 2, dim=-1)
    truth('q_gate_interleaved_per_head', int(pq[0,0,1,0])==512 and int(pg[0,0,0,0])==256)

result = {'passed': all(x['passed'] for x in checks), 'scope': 'exact selected upstream fallback bodies plus independent tiny CPU oracles; not package/model integration',
          'upstream_commit': '4815a0a6a064214f2d8208c094464a5a6b76ca8d', 'source_sha256': hashlib.sha256(raw).hexdigest(),
          'ast_adaptation': 'remove decorators only; no function-body edits; future annotations; explicit torch/F/nn/silu dependencies',
          'definitions': provenance, 'checks': checks, 'gate_counterexample': gate_counterexample,
          'python': sys.version, 'torch': torch.__version__, 'threads': torch.get_num_threads(),
          'device': 'cpu', 'original_model_weights_loaded': False}
(root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'passed': result['passed'], 'checks': len(checks), 'torch': torch.__version__}))
