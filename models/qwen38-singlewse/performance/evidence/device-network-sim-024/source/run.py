"""Capture stopped bulk parser/route state; never substitute for neural evidence."""
import json
from pathlib import Path
import numpy as np
from backend import runtime
from device_client import Encoder,Client
config=json.loads(Path('selection.json').read_text());error=None
with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
print(json.dumps(dict(phase='runtime_enter')),flush=True)
with runtime(False) as (runner,dtype,order):
 encoder=Encoder(config['internal_target_count']);frames=[];indices=[0,7,1,6,2,5,3,4]
 for n in range(61):
  i=indices[n%8];first=(n//8)*256
  frames.append(encoder.command(config['workers'][i]['tag'],0,0,first,256,payload=data['bank_'+str(i)][first:first+256]))
 client=Client(runner,dtype,order,config['gateway'],completion_timeout=5,progress=lambda p:print(json.dumps(p),flush=True))
 try:client.stream(frames)
 except TimeoutError as caught:error=str(caught);print(json.dumps(dict(timeout=error)),flush=True)
from inspect_core import inspect
result=inspect(Path.cwd());result.update(passed=error is None,normal_stop=True,completion_error=error,requested_commands=61)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if error:raise RuntimeError('Incomplete bulk diagnostic: '+error)
