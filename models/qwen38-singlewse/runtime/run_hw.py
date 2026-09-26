"""Physical invocation of the same real-weight numerical driver as simulation."""
import json
from pathlib import Path
import sys
from source_gate import verify
from run import main

verify()
if not json.loads(Path("sram.json").read_text())["passed"]:
    raise ValueError("SRAM gate required before physical allocation")
sys.argv = ["run.py", "--physical"]
main()
verify()
