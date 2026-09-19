"""One finite simulation reusing the independently accepted SDK002 binary files."""
from pathlib import Path
import subprocess
import sys
from reuse_gate import verify_reuse

ROOT = Path(__file__).resolve().parent
verify_reuse(ROOT, auxiliary='absent')
subprocess.run([sys.executable, '-B', 'driver.py'], check=True, timeout=300)
verify_reuse(ROOT, auxiliary='exact')
