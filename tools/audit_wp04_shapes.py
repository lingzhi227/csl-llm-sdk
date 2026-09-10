"""Audit saved model metadata without opening any weight payload."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(saved):
    config = json.loads((saved / 'config.json').read_text())
    text = config['text_config']
    if (text['hidden_size'], text['intermediate_size'], text['num_hidden_layers'],
        text['num_attention_heads'], text['num_key_value_heads'], text['head_dim'],
        text['linear_num_key_heads'], text['linear_num_value_heads'],
        text['linear_key_head_dim'], text['linear_value_head_dim'], text['linear_conv_kernel_dim']) != (
            5120, 17408, 64, 24, 4, 256, 16, 48, 128, 128, 4):
        raise ValueError('Config differs from audited target')
    if text['layer_types'] != ['full_attention' if i % 4 == 3 else 'linear_attention' for i in range(64)]:
        raise ValueError('Layer order differs from audited target')
    headers = json.loads((saved / 'tensor-headers.json').read_text())
    index = json.loads((saved / 'model.safetensors.index.json').read_text())['weight_map']
    actual = {name: (shard, tensor) for shard, tensors in headers.items()
              for name, tensor in tensors.items() if name != '__metadata__'}
    prefix = 'model.language_model.'
    expected = {prefix+'embed_tokens.weight': [248320,5120], prefix+'norm.weight': [5120], 'lm_head.weight': [248320,5120]}
    for i, kind in enumerate(text['layer_types']):
        shapes = {'input_layernorm.weight': [5120], 'post_attention_layernorm.weight': [5120],
                  'mlp.up_proj.weight': [17408,5120], 'mlp.gate_proj.weight': [17408,5120], 'mlp.down_proj.weight': [5120,17408]}
        if kind == 'full_attention':
            shapes.update({'self_attn.q_proj.weight': [12288,5120], 'self_attn.k_proj.weight': [1024,5120],
                           'self_attn.v_proj.weight': [1024,5120], 'self_attn.o_proj.weight': [5120,6144],
                           'self_attn.q_norm.weight': [256], 'self_attn.k_norm.weight': [256]})
        else:
            shapes.update({'linear_attn.in_proj_qkv.weight': [10240,5120], 'linear_attn.in_proj_z.weight': [6144,5120],
                           'linear_attn.in_proj_a.weight': [48,5120], 'linear_attn.in_proj_b.weight': [48,5120],
                           'linear_attn.out_proj.weight': [5120,6144], 'linear_attn.conv1d.weight': [10240,1,4],
                           'linear_attn.A_log': [48], 'linear_attn.dt_bias': [48], 'linear_attn.norm.weight': [128]})
        expected.update({prefix+f'layers.{i}.'+name: shape for name, shape in shapes.items()})
    rows = []
    for name, shape in expected.items():
        shard, tensor = actual[name]
        passed = tensor['shape'] == shape and tensor['dtype'] == 'BF16' and index[name] == shard
        rows.append({'name': name, 'shape': tensor['shape'], 'dtype': tensor['dtype'], 'shard': shard, 'passed': passed})
    extra = [n for n in actual if n.startswith(prefix) and n not in expected]
    return {'passed': all(x['passed'] for x in rows) and not extra, 'matched_tensors': sum(x['passed'] for x in rows),
            'unexpected_text_tensors': extra, 'scope': 'name/shape/dtype/index metadata only; no tensor-content or model execution',
            'sources': {n: hashlib.sha256((saved/n).read_bytes()).hexdigest()
                        for n in ('config.json', 'tensor-headers.json', 'model.safetensors.index.json')}, 'tensors': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.saved_metadata)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise SystemExit('Metadata mismatch')
