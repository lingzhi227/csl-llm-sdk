"""Two finite subprocesses within the already enforced single SDK unit."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-B','compile.py'],check=True,timeout=70)
assert json.loads((ROOT/'compiled.json').read_bytes())['passed']
subprocess.run([sys.executable,'-B','driver.py'],check=True,timeout=30)
