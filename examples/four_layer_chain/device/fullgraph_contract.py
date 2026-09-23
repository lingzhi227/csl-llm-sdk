"""Compose pinned family role/bank contracts in global wafer coordinates."""
from pathlib import Path
import importlib.util,json
ROOT=Path(__file__).resolve().parent
FAMILIES={}
for family in ('linear','attention'):
 root=ROOT/'family-contracts'/family
 spec=importlib.util.spec_from_file_location('qualified_'+family,root/'fullgraph_contract.py')
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 FAMILIES[family]=(module,json.loads((root/'plan.json').read_bytes()),json.loads((root/'transport-map.json').read_bytes()),json.loads((root/'EXPORTS.json').read_bytes())['exports'])
def roles():
 plan=json.loads((ROOT/'lowered.json').read_bytes());result={}
 for region in plan['regions']:
  family=region['family'];module,old,transport,exports=FAMILIES[family]
  for (x,y),p in module.roles(old,transport).items():
   point=x+region['x'],y;assert point not in result
   result[point]=dict(p,family=family,layer_id=region['layer'],request_id=plan['request_id'],stage_id=plan['stage_id'],chain_input=region['input_source']!='host_embedding',chain_output=region['output_target']!='host_observation')
 assert len(result)==119*1160
 return result

def expected_arrays(params):
 module,old,transport,exports=FAMILIES[params['family']]
 result=module.expected_arrays(params,exports)
 result['checkpoint_control']=(32,4)
 if params['role'] in (2,3):
  for name in ('norm.inout','norm.gains','norm.saved'):result[name]=(10240,4)
 if params['role']==2 and params['chain_input'] or params['role']==3 and params['chain_output']:
  result.update({'boundary.buffer':(32,4),'boundary.status':(48,4)})
 return result
