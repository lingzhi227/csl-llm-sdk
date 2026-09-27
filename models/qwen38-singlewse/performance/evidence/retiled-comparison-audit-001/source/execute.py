candidate='retiled-matrix-sim-001'
import json,time
from pathlib import Path
from source_gate import verify
from compare_retiled_matrix import compare
start=time.monotonic()
try:
 verify();runs=Path('/srv/model-storage/qwen38-singlewse/runs')
 result=compare(runs/'full-bf16-matrix-sim-017',runs/candidate);verify()
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
 Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-start,result=result),indent=2)+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
