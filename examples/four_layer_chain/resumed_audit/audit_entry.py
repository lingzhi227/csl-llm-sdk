"""Continue only unfinished layers against immutable original physical evidence."""
import gc,hashlib,io,json,time
from pathlib import Path
from audit_gate import ROOT,BASE,verify

def main():
 started=time.monotonic();data,_,_=verify()
 from runtime_gate import pinned
 from runtime_evidence import Evidence,AuditPrepared,numeric_inputs
 from runtime_prepared import Prepared
 from runtime_families import load
 from runtime_timing import atomic_json
 from qualified_linear_math import load as load_linear
 from qualified_attention_math import load as load_attention
 from linear_numeric import audit as linear
 from attention_numeric import audit as attention
 from preprocess_numerics_v2 import validate
 import numpy as np
 inputs=json.loads((BASE/'runtime-inputs.json').read_bytes());capture=data['values']['capture.json']
 host=json.loads((BASE/'host-plan.json').read_bytes());lowered=json.loads((BASE/'lowered.json').read_bytes())
 prepared=Prepared(inputs['prepared']);families=load()
 oldpin=data['inputs']['baseline_pins']['numeric-layer0.json']
 reports=[dict(layer=0,root=str(BASE),path='numeric-layer0.json',**oldpin,status='all_operator_gates_passed',reused_without_execution=True)]
 if any((ROOT/f'numeric-layer{i}.json').exists() for i in (1,2,3)) or (ROOT/'numeric-audit.json').exists():raise ValueError('One separate resumed audit; no overwrite')
 for layer in (1,2,3):
  pin=inputs['nominal_outputs'][str(layer)];encoded=pinned(Path(inputs['reference_root'])/f'step0-layer{layer}-output.npz',pin,65536)
  with np.load(io.BytesIO(encoded),allow_pickle=False) as archive:
   if archive.files!=['hidden'] or archive['hidden'].shape!=(1,5,5120) or archive['hidden'].dtype!=np.dtype('<u2'):raise ValueError('Exact existing nominal rows')
   nominal=archive['hidden'][0,:2].copy()
  plan,copies=numeric_inputs(families,lowered,host,layer);evidence=Evidence(BASE,capture,host,lowered,layer)
  math=load_linear() if layer<3 else load_attention()
  if layer<3:math['preprocess']=validate
  result=(linear if layer<3 else attention)(evidence,AuditPrepared(prepared,evidence),plan,copies,math,nominal)
  result.update(original_layer=layer,physical_x_offset=plan['physical_x_offset'],nominal_reference=pin,input_source='preceding_original_layer_actual_device_hidden',nominal_path='official_five_token_chunk_prefill; actual device runs sequential recurrent positions',preprocess_domain_version='chain-exp-domain-v2' if layer<3 else 'unchanged_attention',quantitative_bounds_unchanged=True)
  name=f'numeric-layer{layer}.json';atomic_json(ROOT/name,result);raw=(ROOT/name).read_bytes()
  reports.append(dict(layer=layer,root=str(ROOT),path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),status=result['status'],reused_without_execution=False))
  del evidence,math,result;gc.collect()
 preparation=prepared.verify_unchanged();verify()
 result=dict(status='all_four_original_layer_operator_gates_passed',mode='baseline_resumed_audit',layers=reports,baseline_root=str(BASE),baseline_pins=data['inputs']['baseline_pins'],preprocess_domain_version='chain-exp-domain-v2',quantitative_bounds_unchanged=True,actual_inputs_conditioned=True,whole_chain_propagated_enclosure=False,nominal_prefill_vs_recurrent_path_distinction=True,restored_position1_qualified=False,
  inherited_pre_layer0_gates=dict(journal_and_control='Successful predecessor path in exact original runtime_audit.py before committed numeric-layer0.json; not rerun in this resumed audit.',controller_integrity_review_sha256=hashlib.sha256((ROOT/'ORIGINAL-RAW-INTEGRITY-REVIEW.json').read_bytes()).hexdigest(),original_failed_audit_preserved=True),
  preparation=preparation,total_seconds=time.monotonic()-started,SDK_imported=False,CPU_forward=False,full_model=False,controller_acceptance=False)
 atomic_json(ROOT/'numeric-audit.json',result)

if __name__=='__main__':main()
