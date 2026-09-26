"""Audit actual SRAM before invoking the bounded microkernel runtime."""
from pathlib import Path
import runpy
from check_sram import check

check(Path.cwd())
runpy.run_path("run.py", run_name="__main__")
