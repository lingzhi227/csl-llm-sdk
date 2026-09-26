"""Audit every original E8M0 scale against the compiled kernel's FP32 domain."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from checkpoint import Checkpoint

p=argparse.ArgumentParser()
p.add_argument('--model',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
assert (args.model/'COMPLETE.json').exists()
c=Checkpoint(args.model)
records=[]
for name in sorted(c.index):
    if not name.endswith('_scales'):continue
    values=c.tensor(name)
    record=dict(tensor=name,count=int(values.size),minimum=int(values.min()),maximum=int(values.max()),
                outside_kernel_domain=int(np.count_nonzero((values<4)|(values>249))))
    records.append(record)
    del values
result=dict(passed=all(r['outside_kernel_domain']==0 for r in records),tensor_count=len(records),
            total_scales=sum(r['count'] for r in records),records=records)
assert len(records)==48
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if not result['passed']:raise ValueError('Original scale outside compiled kernel domain')
