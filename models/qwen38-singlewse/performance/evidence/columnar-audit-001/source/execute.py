import json,time
from pathlib import Path
from source_gate import verify
from audit_columnar import audit
start=time.monotonic()
try:
 verify();result=audit(Path('.'));verify()
 Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
 Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-start,result=result),indent=2)+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
