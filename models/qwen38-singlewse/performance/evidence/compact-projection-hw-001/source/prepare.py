"""Rebind only metadata; reuse the bit-identical qualified P17 original payload/oracle."""
import hashlib,json
from pathlib import Path
plan=json.loads(Path('region.json').read_text());meta=json.loads(Path('reference-fixture.json').read_text())
assert plan['spec']['output_rows']==48 and plan['spec']['input_columns']==5120
assert plan['spec']['tile_rows']==2 and plan['spec']['tile_columns']==128
assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']=='b9fc3fa2e6150f2de1e017cc5f0490ec4374ca527e25bc1e22e4250e9f646f1c'
for old,new in zip(meta['workers'],plan['workers']):
 for key in ['rank','group','k','fp8_slots','bf16_slots','target_slot']:assert old[key]==new[key]
meta['workers']=plan['workers'];meta['application']=plan['application'];meta['scope']='Complete original48x5120 BF16 matrix on compact2D distributed-ingress projection backend,960 original native tiles and40 input owners. Same P17 fixture bytes and ordered oracle; supplementary representative FP8 smoke only. Host-owned initial operands; not model inference.'
meta['reference_fixture']='full-bf16-matrix-sim-017 / full-bf16-matrix-hw-003'
meta['coordinate_binding']='New component coordinates; capacities and original tile bytes identical to P17, not a global-atlas migration.'
with Path('fixture.json').open('x') as f:json.dump(meta,f,indent=2);f.write('\n')
print(json.dumps(dict(phase='fixture_rebound',fixture_sha256=meta['fixture_sha256'])),flush=True)
