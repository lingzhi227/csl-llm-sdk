"""Gate actual WSE-3 application ELF sections plus a 4 KiB stack allowance."""
import hashlib
import json
from pathlib import Path
from elf_inventory import inventory, admit_wse3_sram


def check(root):
    records = []
    for path in sorted(root.glob("out/bin/*.elf")):
        raw = path.read_bytes()
        gate = admit_wse3_sram(inventory(raw), stack_allowance=4096, ceiling=48128)
        records.append(dict(file=str(path.relative_to(root)), sha256=hashlib.sha256(raw).hexdigest(), **gate))
    if not records:
        raise ValueError("No application ELF files")
    config_path=root/'experiment.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    if config.get('shared_programs'):
        result=dict(passed=all(r['passed'] for r in records),
                    physical_application_pes=config['physical_application_pes'],
                    unique_application_programs=len(records),records=records)
    else:
        result = dict(passed=all(r["passed"] for r in records), application_pes=len(records), records=records)
    (root / "sram.json").write_text(json.dumps(result, indent=2)+"\n")
    if not result["passed"]:
        raise ValueError("Actual SRAM limit failed")
    return result


if __name__ == "__main__":
    print(json.dumps(check(Path.cwd())))
