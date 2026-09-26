"""Inspect retained reference vectors under unchanged v1 thresholds.

This diagnostic collects every violation; it cannot turn a failed criterion
into acceptance and does not re-execute the model or change its observations.
"""
import argparse,ast,hashlib,json,math
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
root=a.reference
tree=ast.parse((root/'run.py').read_text())
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['expand','metrics','selection']],type_ignores=[])
namespace=dict(np=np);exec(compile(functions,'original_frozen_metric_functions','exec'),namespace)
metrics=namespace['metrics'];expand=namespace['expand']
criteria=json.loads((root/'full-acceptance-v1.json').read_text())
progress=json.loads((root/'progress.json').read_text());count=progress['completed_positions']
generation=json.loads((root/'candidate/generation.json').read_text());positions=generation['processed_positions']
shapes={'layer':(positions,64,5120),'final_norm':(positions,5120),'logits':(positions,248320)}
names={'layer':'layers','final_norm':'final_norm','logits':'logits'}
candidate={k:np.memmap(root/'candidate'/(names[k]+'.bf16'),mode='r',dtype='<u2',shape=v) for k,v in shapes.items()}
summary={};failed=[];records=[]
for position in range(count):
 for kind,number in [('layer',64),('final_norm',1),('logits',1)]:
  for index in range(number):
   suffix='layer'+str(index) if kind=='layer' else 'norm' if kind=='final_norm' else 'logits'
   reference=np.load(root/'reference'/f'position{position}-{suffix}.npy',allow_pickle=False)
   observed=candidate[kind][position,index] if kind=='layer' else candidate[kind][position]
   limit=criteria['full_vocabulary_logits'] if kind=='logits' else criteria['per_position_all64_layers_and_final_norm']
   value=metrics(observed,reference,limit);record=dict(position=position,kind=kind,index=index,**value)
   records.append(record);s=summary.setdefault(kind,dict(checks=0,failed=0,max_relative_l2=0,min_cosine=1,max_abs_error=0,max_abs_over_rms=0))
   s['checks']+=1;s['failed']+=int(not value['passed']);s['max_relative_l2']=max(s['max_relative_l2'],value['relative_l2']);s['min_cosine']=min(s['min_cosine'],value['cosine']);s['max_abs_error']=max(s['max_abs_error'],value['max_abs_error']);s['max_abs_over_rms']=max(s['max_abs_over_rms'],value['max_abs_error_over_reference_rms'])
   if not value['passed']:
    left=expand(observed);right=expand(reference);delta=np.abs(left-right);order=np.argsort(delta)[-5:][::-1]
    outliers=[]
    for i in order:
     i=int(i);cb=int(observed[i]);rb=int(reference[i]);rc=rb+1 if right[i]>=0 else rb-1
     ulp=abs(float(expand(np.array([rc],np.uint16))[0])-float(right[i]))
     outliers.append(dict(index=i,candidate=float(left[i]),reference=float(right[i]),candidate_bits=cb,reference_bits=rb,
       absolute_error=float(delta[i]),reference_bf16_ulp=ulp,error_in_reference_ulps=float(delta[i])/ulp if ulp else None))
    failed.append(dict(record,outliers=outliers))
Path('comparisons.json').write_text(json.dumps(records,indent=2)+'\n')
Path('violations.json').write_text(json.dumps(failed,indent=2)+'\n')
result=dict(diagnostic=True,full_model_accepted=False,criteria_changed=False,compared_positions=count,
  expected_full_positions=positions,summary=summary,failed_vector_count=len(failed),first_failure=failed[0] if failed else None,
  source_sha256=hashlib.sha256((root/'run.py').read_bytes()).hexdigest(),
  criteria_sha256=hashlib.sha256((root/'full-acceptance-v1.json').read_bytes()).hexdigest())
Path('COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
