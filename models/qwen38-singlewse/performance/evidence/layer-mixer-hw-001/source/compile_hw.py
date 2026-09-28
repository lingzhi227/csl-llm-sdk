"""Fresh physical compilation of the complete resident original MLP component."""
import hashlib,json,logging,tarfile
from pathlib import Path
from backend import complete_mlp_profile,complete_mixer_profile,file_sha256,message_limit
from bounded_compiler import compiler_class
from check_sram import check
from job_capture import install
from source_gate import verify


def admit(root,artifact,config):
    if not (complete_mlp_profile(config) or complete_mixer_profile(config)) or message_limit(config)!=64<<20:
        raise ValueError('Exact complete original stage profile required')
    flags='--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=67,1 --memcpy --channels=1 --max-parallelism=1 -o out'
    if complete_mixer_profile(config):flags+=' --params=MEMCPYH2D_DATA_1_ID:14,MEMCPYD2H_DATA_1_ID:15'
    if artifact.stat().st_size>64<<20:raise ValueError('Compressed artifact budget')
    with tarfile.open(artifact) as archive:
        members=archive.getmembers()
        if len(members)>16000 or sum(m.size for m in members)>512<<20:
            raise ValueError('Expanded artifact budget')
        apps=[m for m in members if m.isfile() and '/out/bin/' in m.name and m.name.endswith('.elf')]
        if not 1<=len(apps)<=11388:raise ValueError('Image count budget')
        destination=root/'out/bin';destination.mkdir(parents=True);names=set()
        for member in apps:
            name=Path(member.name).name
            # Sparse shared load-rectangle metadata can exceed1MiB even when
            # every PE's live low SRAM is below48KiB. SRAM is checked below.
            if name in names or member.size>2<<20:raise ValueError('Image identity or size')
            names.add(name);(destination/name).write_bytes(archive.extractfile(member).read())
        sources={}
        for m in members:
            if m.isfile() and '/csl/' in m.name and m.name.endswith('.csl'):
                name=Path(m.name).name
                if name in sources:raise ValueError('Duplicate embedded source')
                sources[name]=m
        for source in root.glob('*.csl'):
            if source.name not in sources or archive.extractfile(sources[source.name]).read()!=source.read_bytes():
                raise ValueError('Embedded CSL identity mismatch: '+source.name)
    result=check(root,offset=(67,1))
    if result['application_pes']!=11388:raise ValueError('Whole component coverage')
    Path('artifact.json').write_text(json.dumps(dict(artifact=str(artifact),flags=flags,
        bytes=artifact.stat().st_size,sha256=file_sha256(artifact)),indent=2)+'\n')
    return result


def main():
    verify();install('compile.jobs');logging.basicConfig(level=logging.INFO)
    root=Path.cwd();config=json.loads(Path('experiment.json').read_text())
    if not (complete_mlp_profile(config) or complete_mixer_profile(config)):raise ValueError('Exact complete original stage profile required')
    Compiler=compiler_class(max_artifact_bytes=64<<20)
    flags='--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=67,1 --memcpy --channels=1 --max-parallelism=1 -o out'
    if complete_mixer_profile(config):flags+=' --params=MEMCPYH2D_DATA_1_ID:14,MEMCPYD2H_DATA_1_ID:15'
    with Compiler(disable_version_check=True,resource_cpu=2000,resource_mem=4<<30) as compiler:
        artifact=Path(compiler.compile(str(root),'layout.csl',flags,str(root)))
    admit(root,artifact,config);verify()


if __name__=='__main__':main()
