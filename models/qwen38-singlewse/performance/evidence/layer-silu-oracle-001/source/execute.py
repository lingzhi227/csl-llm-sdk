import json,sys,traceback
from pathlib import Path
from bf16_silu_lut import build_and_prove
import numpy as np
try:
 table,proof=build_and_prove()
 with Path('silu-table.npy').open('xb') as f:np.save(f,table)
 Path('COMPLETE.json').write_text(json.dumps(proof,indent=2)+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
