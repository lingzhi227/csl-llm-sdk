"""Compile original0-3 once; stream the resulting archive for inspection."""
import hashlib,logging,os,subprocess
from pathlib import Path
from source_gate import ROOT,verify,PROFILE
from hw00.job_capture import install
from hw00.store import atomic_json

def main():
 verify();install(os.environ['HW00_JOB_CAPTURE']);logging.basicConfig(level=logging.INFO)
 from cerebras.sdk.client import SdkCompiler
 flags='--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=16 --max-parallelism=1 -o out'
 with SdkCompiler(disable_version_check=True,resource_cpu=PROFILE['worker_cpu_millicores'],resource_mem=PROFILE['worker_memory_bytes']) as compiler:
  from template_guard import verify_template
  verify_template()
  artifact=compiler.compile(str(ROOT),'layout.csl',flags,str(ROOT))
 p=Path(artifact)
 if p.parent.resolve()!=ROOT.resolve() or p.is_symlink() or not p.is_file() or p.stat().st_size>PROFILE['host_file_bytes']:raise ValueError('Artifact path/size bound')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 atomic_json(ROOT/'artifact.json',dict(artifact=str(p),flags=flags,bytes=p.stat().st_size,sha256=h.hexdigest()))
 subprocess.run(['/software/cerebras/cs_sdk-2.10/cs_python','inspect_full_elf.py'],check=True,timeout=120)
 verify()
if __name__=='__main__':main()
