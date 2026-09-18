"""One physical runtime; exact accepted three-input numerical/retention driver."""
import hashlib
import importlib.metadata
import json
import logging
import os
import sys
from source_gate import ROOT, verify
from hw00.job_capture import install
from hw00.store import atomic_json

sys.path.insert(0, str(ROOT / "core"))
from accepted_inputs import load_prepared
from qwen38.resident_sdk_driver import Evidence, NativeAdapter, execute


def main():
    verify()
    artifact = json.loads((ROOT / "artifact.json").read_text())
    from pathlib import Path
    if hashlib.sha256(Path(artifact["artifact"]).read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("Compiled artifact identity changed")
    if not all(json.loads((ROOT / (name + ".json")).read_text())["passed"] for name in ("sram", "placement")):
        raise ValueError("Physical compile gates are incomplete")
    weights, hidden, oracle = load_prepared()  # All inputs checked before allocation.
    install(os.environ["HW00_JOB_CAPTURE"])
    logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkRuntime
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

    class Runtime:
        def __init__(self):
            self.closed = False
            self.runtime = SdkRuntime(artifact["artifact"], simulator=False, disable_version_check=True,
                                      resource_cpu=2000, resource_mem=4 << 30)
            self.runtime.__enter__()
        def __getattr__(self, name):
            return getattr(self.runtime, name)
        def load(self):
            pass  # Appliance context entry already loads and starts the program.
        def run(self):
            pass
        def stop(self):
            if not self.closed:
                self.runtime.__exit__(None, None, None)
                self.closed = True

    evidence = Evidence(ROOT / "device-evidence")
    runner = None
    try:
        runner = Runtime()
        backend = NativeAdapter(runner, MemcpyDataType, MemcpyOrder)
        execute(backend, weights, hidden, oracle, evidence)
    finally:
        if runner is not None:
            runner.stop()
    atomic_json(ROOT / "hardware-result.json", dict(simulator=False, runtime_context_exited=runner.closed,
                sdk=importlib.metadata.version("cerebras-sdk"), appliance=importlib.metadata.version("cerebras-appliance"),
                artifact_sha256=artifact["sha256"], full_model=False, scope="8PE resident 192->128->128 three inputs"))
    verify()


if __name__ == "__main__":
    main()
