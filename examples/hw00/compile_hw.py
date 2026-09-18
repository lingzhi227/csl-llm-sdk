"""One ALCF compile job, followed by local artifact/SRAM/placement checks."""
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from source_gate import ROOT, verify
from hw00.job_capture import install
from hw00.store import atomic_json

sys.path.insert(0, str(ROOT / "core"))
from qwen38.elf_inventory import inventory, admit_wse3_sram


def main():
    verify()
    install(os.environ["HW00_JOB_CAPTURE"])
    logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkCompiler
    flags = "--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out"
    with SdkCompiler(disable_version_check=True, resource_cpu=2000, resource_mem=4 << 30) as compiler:
        artifact = compiler.compile(str(ROOT), "layout.csl", flags, str(ROOT))
    path = Path(artifact)
    if not path.is_file() or path.stat().st_size > 64 << 20:
        raise ValueError("Physical compile artifact exceeds bound")
    atomic_json(ROOT / "artifact.json", dict(artifact=str(path), flags=flags, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    extract = ROOT / "compiled"
    extract.mkdir()
    report = {}
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if len(members) > 512 or sum(m.size for m in members) > 64 << 20:
            raise ValueError("Bounded compile artifact inventory required")
        app = [m for m in members if m.isfile() and "/out/bin/" in m.name and m.name.endswith(".elf")]
        if {Path(m.name).name for m in app} != {f"out_{rank}_0.elf" for rank in range(8)} or len(app) != 8:
            raise ValueError("Eight distinct application program IDs required")
        for member in app:
            data = archive.extractfile(member).read()
            gate = admit_wse3_sram(inventory(data), stack_allowance=4096, ceiling=48128)
            if not gate["passed"]:
                raise ValueError("Physical application SRAM gate failed: " + member.name)
            target = extract / "out/bin" / Path(member.name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            report[member.name] = dict(gate, sha256=hashlib.sha256(data).hexdigest())
        # Bind compiled source to the exact accepted CSL; reject omitted sources.
        source_members = {Path(m.name).name: m for m in members if m.isfile() and "/csl/" in m.name and m.name.endswith(".csl")}
        for source in ROOT.glob("*.csl"):
            if source.name not in source_members or archive.extractfile(source_members[source.name]).read() != source.read_bytes():
                raise ValueError("Compiled CSL source mismatch: " + source.name)
    atomic_json(ROOT / "sram.json", dict(passed=True, application_files=report))
    # SDK native ELF metadata is inspected using the existing ALCF image only;
    # no simulation and no additional physical allocation are made.
    subprocess.run([os.environ["CSL_SDK_PYTHON"], "inspect_hw_elf.py"], check=True, timeout=45)
    verify()


if __name__ == "__main__":
    main()
