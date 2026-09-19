"""Compile selected complete layer3 programs at original coordinates on an appliance worker; no CS3 runtime."""
import hashlib,json,logging,os,subprocess,sys,tarfile
from pathlib import Path
from source_gate import ROOT,verify,PROFILE
from hw00.job_capture import install
from hw00.store import atomic_json
sys.path.insert(0,str(ROOT/'core'))
from qwen38.elf_inventory import inventory,admit_wse3_sram

def main():
    verify();install(os.environ['HW00_JOB_CAPTURE']);logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkCompiler
    flags='--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out'
    with SdkCompiler(disable_version_check=True,resource_cpu=PROFILE['worker_cpu_millicores'],resource_mem=PROFILE['worker_memory_bytes']) as compiler:
        artifact=compiler.compile(str(ROOT),'layout.csl',flags,str(ROOT))
    path=Path(artifact)
    if not path.is_file() or path.stat().st_size>128<<20:raise ValueError('Artifact bound')
    atomic_json(ROOT/'artifact.json',dict(artifact=str(path),flags=flags,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    extract=ROOT/'compiled/out/bin';extract.mkdir(parents=True)
    report={}
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        if len(members)>4096 or sum(m.size for m in members)>128<<20:raise ValueError('Compile inventory budget')
        app=[m for m in members if m.isfile() and '/out/bin/' in m.name and m.name.endswith('.elf')]
        if not 1<=len(app)<=1536:raise ValueError('Bounded application program count')
        for member in app:
            name=Path(member.name).name
            if name in report or member.size>3<<20:raise ValueError('Duplicate or oversized application ELF')
            raw=archive.extractfile(member).read();gate=admit_wse3_sram(inventory(raw),4096,48128)
            (extract/name).write_bytes(raw);report[name]=dict(gate,sha256=hashlib.sha256(raw).hexdigest())
        embedded={Path(m.name).name:m for m in members if m.isfile() and '/csl/' in m.name and m.name.endswith('.csl')}
        for source in ROOT.glob('*.csl'):
            if source.name not in embedded or archive.extractfile(embedded[source.name]).read()!=source.read_bytes():
                raise ValueError('Embedded CSL identity: '+source.name)
    passed=all(item['passed'] for item in report.values())
    atomic_json(ROOT/'sram.json',dict(passed=passed,application_files=report))
    # Preserve actual selected-role mapping and storage even when SRAM fails.
    # Placement/storage success is separate from this strict SRAM gate.
    subprocess.run([os.environ['CSL_INSPECT_PYTHON'],'inspect_full_elf.py'],check=True,timeout=120)
    if not passed:raise ValueError('Selected-program SRAM gate')
    verify()

if __name__=='__main__':main()
