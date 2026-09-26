"""Freeze fresh source snapshots; run bounded, serialized workstation jobs."""
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
BASE = "/srv/model-storage/qwen38-singlewse"


def main():
    p = argparse.ArgumentParser(); p.add_argument("attempt"); p.add_argument("--phase",choices=["compile","prepare","run"],default="compile")
    a = p.parse_args(); assert re.fullmatch(r"[0-9]{3}",a.attempt)
    dest=BASE+"/runs/fp8-tile-sim-"+a.attempt
    if a.phase=="compile":
        files={n:(ROOT/"experiments"/n).read_bytes() for n in ["layout.csl","pe.csl","run.py"]}
        files.update({"fp8_gemv.csl":(ROOT/"csl/fp8_gemv.csl").read_bytes(),"prepare.py":(ROOT/"tools/prepare_fp8_tile.py").read_bytes()})
        files.update({n:(ROOT/"runtime"/n).read_bytes() for n in ["backend.py","check_sram.py","elf_inventory.py","run_micro_sim.py"]})
        files["source-manifest.json"]=(json.dumps({"files":{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+"\n").encode()
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode="w") as archive:
            for n,v in files.items():
                info=tarfile.TarInfo(n);info.size=len(v);archive.addfile(info,io.BytesIO(v))
        subprocess.run(["ssh","workstation","mkdir "+shlex.quote(dest)+" && tar -xf - -C "+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=20)
        out=ROOT/"evidence"/("fp8-tile-sim-"+a.attempt);out.mkdir()
        (out/"source-manifest.json").write_bytes(files["source-manifest.json"])
        command=["/opt/cerebras/sdk/2.10.1/cslc","layout.csl","--arch=wse3","--fabric-dims=8,3","--fabric-offsets=4,1","--memcpy","--channels=1","--max-parallelism=1","-o","out"]
    elif a.phase=="prepare":
        command=["/usr/bin/python3","prepare.py","--model",BASE+"/model","--output",dest+"/fixture"]
    else:
        subprocess.run(["ssh","workstation","cp "+shlex.quote(dest)+"/fixture/fixture.npz "+shlex.quote(dest)+"/fixture/fixture.json "+shlex.quote(dest)+"/"],check=True,timeout=15)
        command=["/opt/cerebras/sdk/2.10.1/cs_python","run_micro_sim.py"]
    unit="qwen38-single-fp8-"+a.phase+"-"+a.attempt
    cmd=["systemd-run","--user","--unit="+unit,"--property=WorkingDirectory="+dest,"--property=MemoryMax=2G","--property=MemorySwapMax=0",
         "--property=CPUAffinity=6 7","--property=TasksMax=128","--property=RuntimeMaxSec=300","--property=TimeoutStopSec=5","--property=LimitCORE=0",
         "--property=StandardOutput=append:"+dest+"/"+a.phase+".log","--property=StandardError=append:"+dest+"/"+a.phase+".log",
         "--setenv=OPENBLAS_NUM_THREADS=1","--setenv=OMP_NUM_THREADS=1","flock","-n","/srv/cerebras-workstation/heavy.lock",*command]
    subprocess.run(["ssh","workstation",shlex.join(cmd)],check=True,timeout=20)
    print(json.dumps(dict(unit=unit,remote=dest,phase=a.phase)))


if __name__=="__main__":main()
