"""Retain the original full-state FP64 gate; freeze the new block-order FP32 oracle.

No candidate CSL is imported. The unchanged recurrence reference creates
reference.npz; original FP8 banks are copied from the frozen P10 fixture.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

reference_meta = json.loads(Path('reference.json').read_text())
require_pin = '1279ce5a8e7311ee04b777c20f861111003cec8e03bf18531abc440785221882'
assert hashlib.sha256(Path('reference.npz').read_bytes()).hexdigest() == reference_meta['fixture_sha256'] == require_pin
with np.load('reference.npz', allow_pickle=False) as f:
    reference = {n: f[n] for n in f.files}
bank_meta = json.loads(Path('banks.json').read_text())
assert hashlib.sha256(Path('banks.npz').read_bytes()).hexdigest() == bank_meta['fixture_sha256']
with np.load('banks.npz', allow_pickle=False) as f:
    banks = {n: f[n] for n in ['weights', 'scales']}
state = np.zeros((128, 128), np.float32)
states, outputs = [], []

def reduce(matrix, vector):
    partials = np.zeros((4, 128), np.float32)
    for k in range(128):
        shard = k // 32
        partials[shard] = (partials[shard].astype(np.float64) + matrix[k].astype(np.float64) * float(vector[k])).astype(np.float32)
    result = partials[3]
    for shard in [2, 1, 0]:
        result = np.float32(partials[shard] + result)
    return result

for packet in reference['packet']:
    q, k, v, decay, beta = packet[:128], packet[128:256], packet[256:384], packet[384], packet[385]
    state = np.float32(state * decay)
    prediction = reduce(state, k)
    delta = np.float32(np.float32(v - prediction) * beta)
    state = (state.astype(np.float64) + k.astype(np.float64)[:, None] * delta.astype(np.float64)).astype(np.float32)
    output = reduce(state, q)
    states.append(state.copy()); outputs.append(output)
states, outputs = np.array(states), np.array(outputs)
state_error = np.abs(states.astype(np.float64) - reference['state'])
output_error = np.abs(outputs.astype(np.float64) - reference['output'])
assert np.all(state_error <= reference['state_bound']) and np.all(output_error <= reference['output_bound'])

weights = np.empty((4, 4, 111 * 128), np.uint16)
scales = np.empty((4, 4, 111), np.float32)
origins = []
for y in range(4):
    for x in range(4):
        source_rank = (y * 4 + x) % 6
        sx, sy = bank_meta['owners'][source_rank]
        weights[y, x] = banks['weights'][sy, sx, :111 * 128]
        scales[y, x] = banks['scales'][sy, sx, :111]
        origins.append(dict(worker=[x, y + 1],source_pe=[sx, sy],source_slots=[0, 111]))

def decode(code):
    code = code.astype(np.int32); exponent = (code & 127) >> 3; mantissa = code & 7
    return np.where(code & 128, -1.0, 1.0) * np.where(exponent == 0, mantissa * 2.0**-9, (1 + mantissa / 8) * np.exp2(exponent - 7))

slots = [0, 56, 110]
dot_inputs = np.zeros((3, 4, 4, 65), np.uint32)
dot_outputs = np.zeros((3, 4, 4, 2), np.float32)
dot_exact = np.zeros((3, 4, 4, 2), np.float64)
dot_bounds = np.zeros_like(dot_exact)
for case, slot in enumerate(slots):
    for y in range(4):
        for x in range(4):
            rank = y * 4 + x; ids = np.arange(128)
            codes = (((ids * 31 + rank * 13 + case) % 127) | (((ids + rank + case) % 2) * 128)).astype(np.uint8)
            half = (((codes.astype(np.uint16) & 127) << 7) | ((codes.astype(np.uint16) & 128) << 8)).astype('<u2')
            activation_scale = np.float32(2.0**(-8 - rank % 3))
            dot_inputs[case, y, x, :64] = half.view('<u4'); dot_inputs[case, y, x, 64] = activation_scale.view(np.uint32)
            wc = weights[y, x].reshape(111, 128)[slot].view(np.uint8).reshape(128, 2)
            w, v = decode(wc), decode(codes)
            acc = np.zeros(2, np.float32)
            for j in range(128):
                acc = (acc.astype(np.float64) + w[j] * v[j]).astype(np.float32)
            dot_outputs[case, y, x] = np.float32(np.float32(acc * activation_scale) * scales[y, x, slot])
            factor = float(activation_scale) * float(scales[y, x, slot])
            dot_exact[case, y, x] = (w.T @ v) * factor
            dot_bounds[case, y, x] = (260 * 2**-24 / (1 - 260 * 2**-24)) * (np.abs(w).T @ np.abs(v)) * factor + np.finfo(np.float32).tiny
assert np.all(np.abs(dot_outputs.astype(np.float64) - dot_exact) <= dot_bounds)
with Path('fixture.npz').open('xb') as f:
    np.savez_compressed(f, weights=weights, scales=scales, packet=reference['packet'], expected_state=states,
                        expected_output=outputs, fp64_state=reference['state'], state_bound=reference['state_bound'],
                        fp64_output=reference['output'], output_bound=reference['output_bound'],
                        dot_input=dot_inputs, dot_output=dot_outputs, dot_exact=dot_exact, dot_bounds=dot_bounds)
meta = dict(application=[4, 5],state_rectangle=[0, 1, 4, 4], fp8_slots=111, batch=8, positions=96,
            simulation_positions=8, simulation_replay=4, physical_replay=8, dot_slots=slots,
            reference_fixture_sha256=require_pin, bank_fixture_sha256=bank_meta['fixture_sha256'], origins=origins,
            ordered_fp32_max_state_error=float(state_error.max()),ordered_fp32_max_output_error=float(output_error.max()),
            fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),
            full_model=False, scope='Complete128x128 recurrent FP32 state on16 atlas32x32 shards cohosted with111 original FP8 tiles each. Preprocessed synthetic Q/K/V/decay/beta from unchanged recurrence fixture. No convolution/gates/norm or full model.',
            criterion='All dependent outputs and batch-terminal full states exactly equal frozen block-order FP32 and satisfy unchanged propagated FP64 bounds. Reset replay exact. Native FP8 dots and full banks retained before/after state updates. Eight autonomous state steps per batch; host supplies the next batch only.')
Path('fixture.json').write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta), flush=True)
