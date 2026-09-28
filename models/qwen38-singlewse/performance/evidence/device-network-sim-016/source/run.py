"""Stop after one commit and inspect compact device state, never neural evidence."""
import json
from pathlib import Path
import numpy as np
from backend import runtime
print(json.dumps(dict(phase='runtime_enter')),flush=True)
with runtime(False) as (runner,dtype,order):
 opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
 command=np.asarray([1,0,0,0,4,0,0x12345678,0xffffffff,0,0x80000001],np.uint32)
 runner.memcpy_h2d(runner.get_id('device_command'),command,2,2,1,1,len(command),**opts)
 print(json.dumps(dict(phase='command_written')),flush=True)
 runner.memcpy_h2d(14,np.asarray([1,len(command),4,1],np.uint32),2,2,1,1,4,**dict(opts,streaming=True))
 print(json.dumps(dict(phase='commit_sent')),flush=True)
observations=[]
for pe,names in [((2,2),['device_submission','device_audit','device_submit_cursor','device_gateway_active','device_words','device_cursor','device_tx_done','device_rx_done']),((0,0),['sys.pending','sys.sending','sys.count','sys.cursor','sys.mode','sys.sequence','sys.have_half','sys.first_half'])]:
 for name in names:
  try:
   value=np.asarray(runner.read_symbol(*pe,name,'uint16')).tolist()
   observations.append(dict(pe=pe,name=name,value=value))
  except Exception as e:observations.append(dict(pe=pe,name=name,error=str(e)))
result=dict(passed=not any('error' in o for o in observations),diagnostic_only=True,physical=False,neural_execution=False,normal_stop=True,observations=observations)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

assert result["passed"], "Diagnostic symbol capture failed"
