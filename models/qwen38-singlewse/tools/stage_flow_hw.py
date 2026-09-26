"""Stage the exact accepted simulator CSL and driver for physical execution."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE="/srv/model-storage/qwen38-singlewse/runs/fp8-flow-sim-001"
DEST="/srv/qwen38-singlewse-hardware/fp8-flow-hw-001"
SESSION=["sh","/path/to/alcf-session.sh","host"]


def main():
    files={n:(ROOT/"experiments/fp8_flow"/n).read_bytes() for n in ["layout.csl","pe.csl","run.py","experiment.json"]}
    for n in ["fp8_block_gemv.csl","fp8_codec.csl","fp8_activation.csl"]:
        files[n]=(ROOT/"csl"/n).read_bytes()
    for name in ["backend.py","bounded_client.py","compile_hw.py","run_hw.py","supervise.py","job_capture.py","source_gate.py","check_sram.py","elf_inventory.py"]:
        files[name]=(ROOT/"runtime"/name).read_bytes()
    for name in ["lifecycle.py","store.py","__init__.py"]:
        files["runtime/"+name]=(ROOT/"runtime"/name).read_bytes()
    names=["layout.csl","pe.csl","fp8_block_gemv.csl","fp8_codec.csl","fp8_activation.csl","run.py","backend.py","experiment.json"]
    check="from pathlib import Path; import json,hashlib; p=Path("+repr(SOURCE)+"); r=json.loads((p/'result.json').read_text()); assert r['passed'] and r['normal_stop']; print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in "+repr(names)+"}))"
    r=subprocess.run(["ssh","workstation","python3 -c "+shlex.quote(check)],check=True,capture_output=True,text=True,timeout=20)
    assert json.loads(r.stdout)=={n:hashlib.sha256(files[n]).hexdigest() for n in names}
    files["source-manifest.json"]=(json.dumps({"files":{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+"\n").encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w") as archive:
        for n,v in files.items():
            info=tarfile.TarInfo(n);info.size=len(v);archive.addfile(info,io.BytesIO(v))
    subprocess.run(SESSION+["mkdir -p /srv/qwen38-singlewse-hardware && mkdir "+shlex.quote(DEST)+" && tar -xf - -C "+shlex.quote(DEST)],input=stream.getvalue(),check=True,timeout=30)
    reader=subprocess.Popen(["ssh","workstation","tar -cf - -C "+shlex.quote(SOURCE)+" fixture.npz fixture.json"],stdout=subprocess.PIPE)
    try:
        subprocess.run(SESSION+["tar -xf - -C "+shlex.quote(DEST)],stdin=reader.stdout,check=True,timeout=30)
    finally:
        reader.stdout.close()
        assert reader.wait(timeout=15)==0
    out=ROOT/"evidence/fp8-flow-hw-001";out.mkdir()
    (out/"staging.json").write_text(json.dumps(dict(remote=DEST,accepted_simulator=SOURCE,
        files=json.loads(files["source-manifest.json"]),physical_dispatched=False),indent=2)+"\n")
    print(DEST)


if __name__=="__main__":main()
