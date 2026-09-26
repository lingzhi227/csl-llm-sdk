"""Post-hoc BF16 backend qualification; never supplies a device operand.

The contract is docs/NUMERICAL-QUALIFICATION.md. Retain all failed checks and
strict-reference differences. A report is accepted only if every check passes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from run_reference import (Checkpoint, ModelConfig, RMSNorm, attention, dense,
                           from_bits, parameter, decode_mxfp4)
from gpt_oss.torch.model import swiglu
from replay_device_states import bits, compare

U = 2.0 ** -24
BF16_EPS = 2.0 ** -7


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def f64(raw):
    return (np.asarray(raw, np.uint32) << 16).view(np.float32).astype(np.float64)


def bf16_exact(value):
    """Direct FP64 to BF16, including ties to even; no double rounding."""
    x = np.asarray(value, np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite mathematical boundary')
    q = np.ldexp(np.ones_like(x), np.maximum(np.frexp(np.abs(x))[1]-8, -133))
    rounded = np.copysign(np.rint(np.abs(x)/q)*q, x).astype(np.float32)
    return (rounded.view(np.uint32) >> 16).astype(np.uint16)


def gamma(n):
    return (n*U)/(1-n*U)


def ordered(bits):
    a = np.asarray(bits, np.int32)
    return np.where(a & 32768, -(a & 32767), a)


def ulp_check(actual, expected, tiny=0.0):
    report = compare(actual, expected)
    distance = np.abs(ordered(actual)-ordered(expected))
    delta = np.abs(f64(actual)-f64(expected))
    valid = (distance <= 1) | (delta <= tiny)
    report.update(passed=bool(report['finite'] and valid.all()),
                  outside=int(np.count_nonzero(~valid)))
    return report


def norm_check(actual, incoming, gain):
    x, g = f64(incoming), f64(gain)
    exact = x / np.sqrt(np.mean(x*x) + 1e-5) * g
    return ulp_check(actual, bf16_exact(exact))


def round_with_bias(lo, hi, bias, before_bias):
    if before_bias:
        lo, hi = f64(bf16_exact(lo)), f64(bf16_exact(hi))
    lo, hi = lo+bias, hi+bias
    # Conservative normal FP32 addition error. BF16 operands are exact FP32.
    e = gamma(1)*np.maximum(np.abs(lo), np.abs(hi)) + np.finfo(np.float32).tiny
    return f64(bf16_exact(lo-e)), f64(bf16_exact(hi+e))


def dot_interval(actual, weights, incoming, bias, chain_depth, before_bias):
    x = f64(incoming)
    w = np.asarray(weights, np.float64)
    exact = w @ x
    error = gamma(chain_depth)*(np.abs(w) @ np.abs(x))
    lo, hi = round_with_bias(exact-error, exact+error, f64(bias), before_bias)
    value = f64(actual)
    finite = bool(np.isfinite(value).all())
    outside = (value < lo) | (value > hi)
    return dict(passed=bool(finite and not outside.any()), finite=finite,
                values=int(value.size), outside=int(outside.sum()),
                max_interval_violation=float(np.max(np.maximum(np.maximum(lo-value, value-hi), 0))),
                max_accumulation_bound=float(error.max()))


def expert_matrix(c, layer, expert, kind, incoming, actual):
    prefix = f'model.layers.{layer}.mlp.experts.{kind}_proj'
    blocks, scales = c.tensor(prefix+'_blocks'), c.tensor(prefix+'_scales')
    bias = c.tensor(prefix+'_bias')[expert]
    records = []
    for start in range(0, actual.size, 160):
        stop = min(start+160, actual.size)
        w = decode_mxfp4(blocks[expert, start:stop], scales[expert, start:stop]).reshape(stop-start, 2880)
        # All original expert weights must be exactly representable as BF16.
        if not np.array_equal(w.astype(np.float64), f64(bf16_exact(w))):
            raise ValueError('MXFP4 weight is not exactly BF16 representable')
        records.append(dot_interval(actual[start:stop], w, incoming,
                                    bias[start:stop], 297, True))
    return dict(passed=all(r['passed'] for r in records),
                values=sum(r['values'] for r in records),
                outside=sum(r['outside'] for r in records),
                max_interval_violation=max(r['max_interval_violation'] for r in records),
                max_accumulation_bound=max(r['max_accumulation_bound'] for r in records))


def join_check(actual, experts, probabilities, residual):
    products = f64(experts)*f64(probabilities)[:, None]
    exact = products.sum(axis=0)
    error = gamma(4)*np.abs(products).sum(axis=0)
    lo, hi = round_with_bias(exact-error, exact+error, f64(residual), True)
    value = f64(actual)
    outside = (value < lo) | (value > hi)
    return dict(passed=bool(np.isfinite(value).all() and not outside.any()),
                values=int(value.size), outside=int(outside.sum()),
                max_interval_violation=float(np.max(np.maximum(np.maximum(lo-value, value-hi), 0))))


def attention_check(actual, expected, residual):
    report = compare(actual, expected)
    delta = f64(actual)-f64(expected)
    update = f64(expected)-f64(residual)
    l2_bound = BF16_EPS*np.linalg.norm(update)
    max_bound = BF16_EPS*max(np.max(np.abs(f64(expected))), np.max(np.abs(update)))
    report.update(passed=bool(report['finite'] and np.linalg.norm(delta) <= l2_bound
                              and np.max(np.abs(delta)) <= max_bound),
                  l2_error=float(np.linalg.norm(delta)), l2_bound=float(l2_bound),
                  max_abs_bound=float(max_bound),
                  relative_update_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(update), 1e-300)))
    return report


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((a.model/'COMPLETE.json').read_text())
    assert receipt['revision'] == '6cee5e81ee83917806bbde320786a8fb61efebee'
    physical = json.loads((a.capture/'PHYSICAL_CAPTURE_COMPLETE.json').read_text())
    result = json.loads((a.capture/'result.json').read_text())
    assert physical['all_owned_jobs_released'] and physical['tokens'] == [13225, 11, 5922]
    assert all(result[k] for k in ('physical', 'normal_stop', 'full_model',
                                  'weights_uploaded_once', 'all_weights_retained', 'west_to_east'))
    assert not result['host_intermediate_neural_computation']
    assert all(r['token_exact'] and r['every_active_pe_completed'] for r in result['records'])
    captures = []
    required = dict(attention=(24,2880), output=(24,2880), ids=(24,4),
                    attention_norm=(24,2880), moe_norm=(24,2880),
                    router_logits=(24,32), router_probabilities=(24,4),
                    expert_gate=(24,4,5760), expert_activation=(24,4,2880),
                    expert_output=(24,4,2880))
    for step in range(2):
        with np.load(a.capture/f'actual-{step+1}.npz', allow_pickle=False) as archive:
            data = {k:archive[k] for k in required}
        for key, shape in required.items():
            assert data[key].shape == shape and data[key].dtype == np.uint16
            if key != 'ids': assert np.isfinite(f64(data[key])).all()
        captures.append(data)
    c = Checkpoint(a.model); config = ModelConfig(num_hidden_layers=24, num_experts=32)
    embeddings = [np.array(c.row_slice('model.embed_tokens.weight', t)[0]) for t in physical['tokens'][:-1]]
    records = []; started = time.monotonic()
    for layer in range(24):
        prefix = f'model.layers.{layer}.'
        incoming = np.stack([embeddings[s] if layer == 0 else captures[s]['output'][layer-1] for s in range(2)])
        module = attention(c, config, layer)
        normalized = from_bits(np.stack([d['attention_norm'][layer] for d in captures]))
        # Validate norm separately, then isolate the original attention equations
        # at the actual device boundary. All prefix K/V come from actual states.
        hook = module.norm.register_forward_hook(lambda module, args, output: normalized)
        expected_attention = bits(module(from_bits(incoming)))
        hook.remove(); del module
        for step, data in enumerate(captures):
            checks = {}
            checks['attention_norm'] = norm_check(data['attention_norm'][layer], incoming[step], c.tensor(prefix+'input_layernorm.weight'))
            checks['attention'] = attention_check(data['attention'][layer], expected_attention[step], incoming[step])
            checks['moe_norm'] = norm_check(data['moe_norm'][layer], data['attention'][layer], c.tensor(prefix+'post_attention_layernorm.weight'))
            checks['router_logits'] = dot_interval(data['router_logits'][layer], f64(c.tensor(prefix+'mlp.router.weight')),
                data['moe_norm'][layer], c.tensor(prefix+'mlp.router.bias'), 125, False)
            scores = f64(data['router_logits'][layer])
            canonical = np.lexsort((np.arange(32), -scores))[:4]
            ids = data['ids'][layer]
            checks['router_ids'] = dict(passed=bool(np.array_equal(ids, canonical)),
                actual=ids.tolist(), canonical=canonical.tolist(),
                fourth_rank_tied_ids=np.flatnonzero(scores == scores[canonical[3]]).tolist())
            probabilities = bits(torch.softmax(from_bits(data['router_logits'][layer][canonical]), dim=0))
            checks['router_probabilities'] = ulp_check(data['router_probabilities'][layer], probabilities)
            experts = []
            for rank, expert in enumerate(ids):
                gate = expert_matrix(c, layer, int(expert), 'gate_up', data['moe_norm'][layer], data['expert_gate'][layer,rank])
                expected_activation = bits(swiglu(from_bits(data['expert_gate'][layer,rank])))
                activation = ulp_check(data['expert_activation'][layer,rank], expected_activation, tiny=1e-32)
                down = expert_matrix(c, layer, int(expert), 'down', data['expert_activation'][layer,rank], data['expert_output'][layer,rank])
                experts.append(dict(expert=int(expert), rank=rank, gate_up=gate, swiglu=activation, down=down,
                                    passed=gate['passed'] and activation['passed'] and down['passed']))
            checks['experts'] = dict(passed=all(e['passed'] for e in experts), records=experts)
            checks['join_residual'] = join_check(data['output'][layer], data['expert_output'][layer],
                                                 data['router_probabilities'][layer], data['attention'][layer])
            record = dict(step=step, layer=layer, checks=checks, passed=all(v['passed'] for v in checks.values()))
            records.append(record)
            with (a.output/'layers.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
            print(json.dumps(dict(step=step, layer=layer, passed=record['passed'],
                failed=[k for k,v in checks.items() if not v['passed']], seconds=time.monotonic()-started)), flush=True)
    norm = RMSNorm(2880, device='meta')
    norm.scale = parameter(dense(c, 'model.norm.weight').float())
    hidden = norm(from_bits(np.stack([d['output'][-1] for d in captures])))
    head = dense(c, 'lm_head.weight')
    logits = torch.nn.functional.linear(hidden, head); del head
    best = torch.topk(logits, 8, dim=-1)
    head_check = dict(passed=best.indices[:,0].tolist() == physical['tokens'][1:],
                      top_ids=best.indices.tolist(), top_logits=best.values.float().tolist())
    bound_files = ['PHYSICAL_CAPTURE_COMPLETE.json','result.json','source-manifest.json',
                   'PRE_RUN_ADMISSION.json','weight-load.json','weight-retention.json','actual-1.npz','actual-2.npz']
    report = dict(passed=all(r['passed'] for r in records) and head_check['passed'],
        scope='two-token physical integration with actual-input BF16 operator qualification; no corpus or long-context claim',
        validation_cpu_only=True, physical_tokens=physical['tokens'],
        physical_evidence={n:digest(a.capture/n) for n in bound_files},
        contract_sha256=digest(a.contract), verifier_sha256=digest(Path(__file__)),
        checkpoint_revision=receipt['revision'], checkpoint_receipt_sha256=digest(a.model/'COMPLETE.json'),
        strict_original_reference_passed=result['strict_original_reference_passed'],
        head=head_check, layers=records, seconds=time.monotonic()-started)
    (a.output/'QUALIFICATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'layers'}), flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
