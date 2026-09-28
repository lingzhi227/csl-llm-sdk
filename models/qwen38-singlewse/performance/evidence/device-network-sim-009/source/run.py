"""Bounded diagnostic only: inspect actual sentinel state after the SDK RPC."""
import json
import numpy as np
from backend import runtime
print(json.dumps(dict(phase='runtime_enter')),flush=True)
with runtime(False) as (runner,dtype,order):
 print(json.dumps(dict(phase='runtime_ready')),flush=True)
 runner.launch('sentinel_ping',np.uint32(0xf1230001),nonblock=False)
 print(json.dumps(dict(phase='rpc_sent')),flush=True)
 runner.dump_core('diagnostic.core')
 print(json.dumps(dict(phase='core_saved')),flush=True)
 for y in range(3):
  for x in (3,7):
   try:
    result=runner.read_symbol(x,y if x==3 else y+1,'sentinel','uint32')
    print(json.dumps(dict(phase='debug_symbol',pe=[x,y if x==3 else y+1],value=np.asarray(result).tolist())),flush=True)
   except Exception as e:print(json.dumps(dict(phase='debug_error',pe=[x,y],error=str(e))),flush=True)
 result=np.zeros(2,np.uint32)
 print(json.dumps(dict(phase='sentinel_read_one')),flush=True)
 runner.memcpy_d2h(result,runner.get_id('sentinel'),3,0,1,1,2,streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
 print(json.dumps(dict(phase='sentinel_result',value=result.tolist())),flush=True)
