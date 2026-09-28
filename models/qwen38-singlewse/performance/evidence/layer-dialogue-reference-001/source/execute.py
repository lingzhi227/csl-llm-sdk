import json,time
from pathlib import Path
from source_gate import verify
from prepare import main
started=time.monotonic()
try:
 verify();main();verify()
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,seconds=time.monotonic()-started,proof=json.loads(Path('bank-remap-proof.json').read_text())))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
