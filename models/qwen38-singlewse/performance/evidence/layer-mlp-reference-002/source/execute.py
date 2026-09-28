import json
from pathlib import Path
from source_gate import verify
try:
 verify()
 from prepare_layer_norm import main
 main();verify()
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
