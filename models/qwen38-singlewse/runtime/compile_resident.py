"""Compile and stream-audit the exact complete candidate layout; no inference."""
import json,logging,shutil
from pathlib import Path
from source_gate import verify
from job_capture import install
from bounded_compiler import compiler_class
from artifact_stream import audit
from resident_plan import load

def main():
    root=Path.cwd();verify();install('compile.jobs');logging.basicConfig(level=logging.INFO)
    config=json.loads((root/'experiment.json').read_text())
    assert config['full_resident'] and config['application']==[750,1160] and config['application_pes']==870000
    plan=load(root);source=json.loads((root/'full-layout-source.json').read_text())
    assert plan['role_sha256']==source['role_sha256'] and plan['config_sha256']==source['config_sha256']
    assert shutil.disk_usage(root).free>96<<30
    flags='--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out'
    with compiler_class()(disable_version_check=True,resource_cpu=4000,resource_mem=16<<30) as compiler:
        artifact=Path(compiler.compile(str(root),'layout.csl',flags,str(root)))
    result=audit(artifact,root,870000,application=(750,1160),max_elf_bytes=8<<20)
    (root/'sram.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'artifact.json').write_text(json.dumps(dict(artifact=str(artifact),flags=flags,sha256=result['artifact_sha256']),indent=2)+'\n')
    verify()

if __name__=='__main__':main()
