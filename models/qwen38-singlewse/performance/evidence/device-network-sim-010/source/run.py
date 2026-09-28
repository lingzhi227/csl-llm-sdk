"""Bounded diagnostic only: distinguish row-specific SDK routes."""
import json
from pathlib import Path
import numpy as np
from backend import runtime
print(json.dumps(dict(phase='runtime_enter')),flush=True)
with runtime(False) as (runner,dtype,order):
 print(json.dumps(dict(phase='runtime_ready')),flush=True)
 runner.launch('sentinel_ping',np.uint32(0xf1230001),nonblock=False)
 print(json.dumps(dict(phase='rpc_sent')),flush=True)
 for y in range(3):
  result=np.zeros(2,np.uint32)
  print(json.dumps(dict(phase='sentinel_read_one',row=y)),flush=True)
  runner.memcpy_d2h(result,runner.get_id('sentinel'),3,y,1,1,2,streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
  print(json.dumps(dict(phase='sentinel_result',row=y,value=result.tolist())),flush=True)
  np.testing.assert_array_equal(result,[0xf1230001,1])
 result=np.zeros(6,np.uint32)
 print(json.dumps(dict(phase='sentinel_read_three')),flush=True)
 runner.memcpy_d2h(result,runner.get_id('sentinel'),3,0,1,3,2,streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
 np.testing.assert_array_equal(result,np.tile([0xf1230001,1],3))
Path('result.json').write_text(json.dumps(dict(passed=True,diagnostic_only=True,neural_execution=False,physical=False,normal_stop=True))+'\n')
