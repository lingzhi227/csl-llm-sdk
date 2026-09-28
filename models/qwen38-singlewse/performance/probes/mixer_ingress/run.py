"""Run the actual original and ingress-fused CSL arithmetic on one simulated PE."""
import hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime

meta=json.loads(Path('fixture.json').read_text())
assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
records=[];capture={}
with runtime(False) as (runner,dtype,order):
 ids={n:runner.get_id(n) for n in ['bank','input','packet','setup','output','audit']}
 opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
 def write(name,value,bits=32):
  value=np.asarray(value,np.uint32).reshape(-1)
  runner.memcpy_h2d(ids[name],value,0,0,1,1,len(value),data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
 def read(name,n,bits=32):
  value=np.zeros(n,np.uint32)
  runner.memcpy_d2h(value,ids[name],0,0,1,1,n,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts);return value
 def canon(x):return np.where(x==0,np.float32(0),x).astype(np.float32).view(np.uint32)
 np.testing.assert_array_equal(read('audit',4),0)
 for count,c in enumerate(meta['cases'],1):
  suffix=str(c['case']);owner=str(c['owner_rank']);bank=data['bank_'+owner];setup=data['setup_'+owner]
  write('bank',bank);write('setup',setup,16);write('input',data['input_'+suffix])
  runner.launch('measure',np.uint32(c['invocation']),np.uint16(c['group']),np.uint16(c['parts']),nonblock=False)
  result=read('output',20).view(np.float32);expected=data['expected_'+suffix]
  assert np.isfinite(result).all()
  np.testing.assert_array_equal(canon(result[:10]),canon(result[10:]))
  np.testing.assert_array_equal(canon(result[:10]),canon(expected))
  np.testing.assert_array_equal(read('packet',133),data['packet_'+suffix])
  np.testing.assert_array_equal(read('bank',323),bank);np.testing.assert_array_equal(read('setup',40,16),setup)
  np.testing.assert_array_equal(read('input',64),data['input_'+suffix])
  np.testing.assert_array_equal(read('audit',4),[c['invocation'],c['group'],c['parts'],count])
  records.append(dict(c,exact=True));capture['output_'+suffix]=result
  print(json.dumps(records[-1]),flush=True)
np.savez('actual.npz',**capture)
Path('result.json').write_text(json.dumps(dict(passed=True,physical=False,normal_stop=True,cases=records,
 original_weight_sample=True,exact_original_packets=True,exact_native_codes=True,exact_baseline_and_independent_outputs=True,
 weights_inputs_setups_retained=True,fixture_sha256=meta['fixture_sha256'],full_matrix=False,full_mixer=False,
 full_stage_admission=False,full_model_speed_target_achieved=False),indent=2)+'\n')
