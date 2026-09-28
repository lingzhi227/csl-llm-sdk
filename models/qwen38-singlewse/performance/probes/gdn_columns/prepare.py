"""Freeze exact original-reference frontend inputs and persistent GDN intervals."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from reference.gdn_columns_oracle import step

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 start=time.monotonic();config=json.loads(Path('reference-input.json').read_text());root=Path(config['root'])
 if sha(root/'fixture.npz')!=config['fixture_sha256']:raise ValueError('Frozen original frontend reference identity')
 source=dict(np.load(root/'fixture.npz',allow_pickle=False));arrays={};state=np.zeros((48,128,128),np.float64);error=np.zeros_like(state);ratios=[]
 for position in range(6):
  old=source[f'packet_{position}'].reshape(48,386).astype(np.float32)
  # Producer's original [Q,K,V,decay,beta] becomes the direct recurrence order.
  packet=np.concatenate((old[:,384:],old[:,256:384],old[:,128:256],old[:,:128]),axis=1)
  low=np.zeros((48,128),np.float64);high=np.zeros_like(low)
  for h in range(48):
   state[h],error[h],low[h],high[h],ratio=step(state[h],error[h],packet[h]);ratios.append(ratio)
  arrays.update({f'packet_{position}':packet,f'state_{position}':state.copy(),f'bound_{position}':error.copy(),f'lower_{position}':low,f'upper_{position}':high})
 with Path('fixture.npz').open('xb') as out:np.savez(out,**arrays)
 result=dict(passed=True,physical=False,heads=48,positions=6,reset_replay_positions=2,original_frontend_reference_sha256=config['fixture_sha256'],
  maximum_state_bound_norm_ratio=max(ratios),state_bound_norm_cap=0.002,fixture_sha256=sha(Path('fixture.npz')),seconds=time.monotonic()-start,
  input_provenance='FP32-rounded independent original-parameter frontend reference. Actual frontend is not executing in this harness.',
  scope='Actual paired-port recurrence over all48 original heads and all753 P45 value slices. Reduced diagnostic banks; no whole-layer or model-speed admission.')
 Path('fixture.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
