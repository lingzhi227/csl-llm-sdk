"""Select original projected inputs; never prepare injected recurrent results."""
import hashlib,json
from pathlib import Path
import numpy as np

def main():
 config=json.loads(Path('reference-input.json').read_text());root=Path(config['root'])
 if hashlib.sha256((root/'fixture.npz').read_bytes()).hexdigest()!=config['fixture_sha256']:raise ValueError('Original reference identity')
 original=dict(np.load(root/'fixture.npz',allow_pickle=False));plan=json.loads(Path('probe-plan.json').read_text());groups=plan['groups']
 data={n:original[n][groups] for n in ['weights','gain','parameters']}
 for position in range(6):
  for name in ['packet','bound','history']:data[f'{name}_{position}']=original[f'{name}_{position}'][groups]
  z=np.zeros((len(groups),384),np.uint16)
  for x,g in enumerate(groups):
   chosen=[]
   for frame in original[f'projected_{position}_{g}']:
    token,row,count,packed,matrix=map(int,frame)
    wanted=(matrix==0 and (128*g<=row<128*g+128 or 2048+128*g<=row<2048+128*g+128 or 4096+384*g<=row<4096+384*g+384)) or (matrix==1 and 384*g<=row<384*g+384) or (matrix in (2,3) and 3*g<=row<3*g+3)
    if wanted:
     chosen.append(frame)
     if matrix==1:z[x,row-384*g:row-384*g+2]=[packed&65535,packed>>16]
   if len(chosen)!=518:raise ValueError('Original group projection coverage')
   data[f'projected_{position}_{x}']=np.array(chosen,np.uint32)
  data[f'z_{position}']=z
 with Path('fixture.npz').open('xb') as out:np.savez(out,**data)
 result=dict(passed=True,groups=groups,positions=6,reset_replay_positions=2,original_fixture_sha256=config['fixture_sha256'],
             fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),injected_recurrent_results=False,
             criterion='A priori frontend interval, independent FP64 recurrence conditioned on observed device frontend inputs, exact history, gated-output interval and bitwise reset replay. Observed values are never sent back to the device.')
 Path('fixture.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
