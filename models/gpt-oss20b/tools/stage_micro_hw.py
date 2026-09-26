"""Stage small sources and relay only the verified tile fixture to ALCF."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SESSION = ["sh", "/path/to/alcf-session.sh", "host"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--kind", choices=["tile","flow","math","chain","transformer","router"], default="tile")
    a = p.parse_args()
    stem={"math":"moe-math","transformer":"transformer-math","router":"router"}.get(a.kind,"mxfp4-"+a.kind)
    if not re.fullmatch(stem+r"-hw-[0-9]{3}", a.name):
        raise ValueError("Named fresh experiment required")
    dest = "/srv/gpt-oss20b-hardware/" + a.name
    mapping = {
        "layout.csl": "experiments/mxfp4_tile/layout.csl",
        "pe.csl": "experiments/mxfp4_tile/pe.csl",
        "mxfp4_gemv.csl": "csl/mxfp4_gemv.csl",
        "run.py": "experiments/mxfp4_tile/run.py",
        "compile_hw.py": "runtime/compile_micro_hw.py",
        "run_hw.py": "runtime/run_micro_hw.py",
        "supervise.py": "runtime/supervise_micro_hw.py",
        "job_capture.py": "runtime/job_capture.py",
        "source_gate.py": "runtime/source_gate.py",
        "runtime/__init__.py": "runtime/__init__.py",
        "runtime/lifecycle.py": "runtime/lifecycle.py",
        "runtime/store.py": "runtime/store.py",
        "elf_inventory.py": "tools/elf_inventory.py",
        "check_sram.py": "tools/check_sram.py",
    }
    if a.kind=="flow":
        mapping.update({"layout.csl":"experiments/mxfp4_flow/layout.csl",
                        "pe.csl":"experiments/mxfp4_flow/pe.csl",
                        "experiment.json":"experiments/mxfp4_flow/experiment.json"})
    if a.kind=="math":
        del mapping['mxfp4_gemv.csl']
        mapping.update({n:'experiments/moe_math/'+n for n in ['layout.csl','pe.csl','run.py']})
        mapping.update({'moe_math.csl':'csl/moe_math.csl','backend.py':'runtime/backend.py'})
    if a.kind in ("chain","transformer"):
        folder="mxfp4_chain" if a.kind=="chain" else "transformer_math"
        mapping.update({n:"experiments/"+folder+"/"+n for n in ("layout.csl","pe.csl","run.py","experiment.json")})
        mapping['backend.py']='runtime/backend.py'
    if a.kind=="transformer":
        del mapping['mxfp4_gemv.csl']
        mapping.update({n:'csl/'+n for n in ['moe_math.csl','rotary_trig.csl','normalization_rope.csl','attention96.csl']})
    if a.kind=="router":
        del mapping['mxfp4_gemv.csl']
        mapping.update({n:'experiments/router/'+n for n in ['layout.csl','pe.csl','run.py','experiment.json']})
        mapping.update({n:'csl/'+n for n in ['bf16_gemv.csl','moe_math.csl','rotary_trig.csl','normalization_rope.csl']})
        mapping['backend.py']='runtime/backend.py'
    files = {name: (ROOT / source).read_bytes() for name, source in mapping.items()}
    attempt={"tile":"002","flow":"002","math":"003","chain":"004","transformer":"001","router":"001"}[a.kind]
    source = "/srv/model-storage/gpt-oss20b/runs/"+stem+"-sim-"+attempt
    names=[n for n in files if n.endswith('.csl') or n in ('run.py','backend.py','experiment.json')]
    check = "import json,pathlib,hashlib; p=pathlib.Path("+repr(source)+"); assert json.loads((p/'result.json').read_text())['passed']; print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in "+repr(names)+"}))"
    accepted=subprocess.run(["ssh","workstation","python3 -c "+shlex.quote(check)],check=True,capture_output=True,text=True,timeout=15)
    if json.loads(accepted.stdout)!={n:hashlib.sha256(files[n]).hexdigest() for n in names}:
        raise ValueError("Source differs from the accepted simulator run")
    manifest = {"files": {n: hashlib.sha256(raw).hexdigest() for n, raw in files.items()}}
    files["source-manifest.json"] = (json.dumps(manifest, indent=2)+"\n").encode()
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for name, raw in files.items():
            entry = tarfile.TarInfo(name)
            entry.size, entry.mode = len(raw), 0o600
            archive.addfile(entry, io.BytesIO(raw))
    command = f"mkdir -p /srv/gpt-oss20b-hardware && mkdir {shlex.quote(dest)} && tar -xf - -C {shlex.quote(dest)}"
    subprocess.run(SESSION+[command], input=payload.getvalue(), check=True, timeout=30)
    fixture={"math":"math-fixture","transformer":"transformer-fixture","router":"router-fixture"}.get(a.kind,"fixture")
    fixture_names=" "+fixture+".npz "+fixture+".json"
    reader = subprocess.Popen(["ssh", "workstation", "tar -cf - -C "+shlex.quote(source)+fixture_names], stdout=subprocess.PIPE)
    try:
        subprocess.run(SESSION+["tar -xf - -C "+shlex.quote(dest)], stdin=reader.stdout, check=True, timeout=30)
    finally:
        reader.stdout.close()
        if reader.wait(timeout=15):
            raise RuntimeError("Fixture relay source failed")
    verify = "cd "+shlex.quote(dest)+" && /opt/cerebras/venv/bin/python -c 'from source_gate import verify; verify(); print(\"STAGED\")'"
    subprocess.run(SESSION+[verify], check=True, timeout=30)
    receipt = dict(remote=dest, sources=manifest, fixture_relayed_without_local_payload_file=True,
                   submitted_hardware_job=False)
    local = ROOT / "evidence" / a.name
    local.mkdir(parents=True, exist_ok=False)
    (local / "staging.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({k:v for k,v in receipt.items() if k!="sources"}))


if __name__ == "__main__":
    main()
