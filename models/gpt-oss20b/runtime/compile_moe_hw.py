"""Bounded physical compilation of the shared-program 31,320-PE MoE layer."""
import hashlib,json,logging,tarfile
from pathlib import Path
from job_capture import install
from source_gate import verify
from check_sram import check

def main():
    verify();install('compile.jobs');logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkCompiler
    root=Path.cwd();config=json.loads((root/'experiment.json').read_text())
    assert config['shared_programs']
    assert ((config.get('full_moe_layer') and config['physical_application_pes']==31320)
            or (config.get('full_attention') and config['physical_application_pes']==4640))
    expected=config['expected_unique_programs'];assert 1<=expected<=512
    flags='--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out'
    with SdkCompiler(disable_version_check=True,resource_cpu=2000,resource_mem=8<<30) as compiler:
        artifact=Path(compiler.compile(str(root),'layout.csl',flags,str(root)))
    if artifact.stat().st_size>128<<20:raise ValueError('MoE artifact exceeds explicit 128 MiB profile')
    with tarfile.open(artifact) as archive:
        members=archive.getmembers()
        if len(members)>4096 or sum(m.size for m in members)>128<<20:raise ValueError('MoE archive exceeds bound')
        apps=[m for m in members if m.isfile() and '/out/bin/' in m.name and m.name.endswith('.elf')]
        if len(apps)!=expected:raise ValueError('Shared program count differs from local full geometry')
        dest=root/'out/bin';dest.mkdir(parents=True)
        for member in apps:(dest/Path(member.name).name).write_bytes(archive.extractfile(member).read())
        sources={Path(m.name).name:m for m in members if m.isfile() and '/csl/' in m.name and m.name.endswith('.csl')}
        for source in root.glob('*.csl'):
            if source.name not in sources or archive.extractfile(sources[source.name]).read()!=source.read_bytes():
                raise ValueError('Embedded source differs from frozen source')
    check(root)
    (root/'artifact.json').write_text(json.dumps(dict(artifact=str(artifact),flags=flags,
      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()),indent=2)+'\n')
    verify()

if __name__=='__main__':main()
