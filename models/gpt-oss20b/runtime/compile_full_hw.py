"""Full physical fabric compilation; all application programs audited before run."""
import hashlib,json,logging,tarfile
from pathlib import Path
from job_capture import install
from source_gate import verify
from check_sram import check

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    verify();logging.basicConfig(level=logging.INFO)
    root=Path.cwd();config=json.loads((root/'experiment.json').read_text())
    assert config['full_model'] and config['shared_programs'] and config['physical_application_pes']==870000
    expected=config['expected_unique_programs']
    if expected is None:
        assert config.get('cluster_compile_first') or config.get('recover_artifact')
    else:assert 1<=expected<=2048
    flags='--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=2 -o out'
    recovery=config.get('recover_artifact')
    if recovery:
        artifact=Path(recovery['artifact'])
        assert digest(artifact)==recovery['sha256']
        audit=recovery['audit']
        assert audit['jobs'] and not audit['error'] and not audit['cleanup_errors']
        assert all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in audit['jobs'])
    else:
        from cerebras.sdk.client import SdkCompiler
        install('compile.jobs')
        with SdkCompiler(disable_version_check=True,resource_cpu=2000,resource_mem=8<<30) as compiler:
            artifact=Path(compiler.compile(str(root),'layout.csl',flags,str(root)))
    limit=512<<20
    if artifact.stat().st_size>limit:raise ValueError('Full artifact exceeds 512 MiB profile')
    with tarfile.open(artifact) as archive:
        members=archive.getmembers()
        # The 2009 full-model ELFs contain 1.216 GB of debug information and
        # executable data, despite the complete compressed archive being 65 MB.
        # Read/extract one bounded ELF at a time; file bytes are not PE SRAM.
        if len(members)>16384 or sum(m.size for m in members)>2<<30:raise ValueError('Full artifact exceeds 2 GiB unpacked archive bound')
        apps=[m for m in members if m.isfile() and '/out/bin/' in m.name and m.name.endswith('.elf')]
        if not 1<=len(apps)<=2048:raise ValueError('Full-model shared program count exceeds admitted bounds')
        if expected is not None and len(apps)!=expected:raise ValueError('Shared program count differs from actual local full compilation')
        dest=root/'out/bin';dest.mkdir(parents=True)
        if len({Path(m.name).name for m in apps})!=len(apps):raise ValueError('Duplicate application ELF names')
        for member in apps:
            if member.size>8<<20:raise ValueError('Individual ELF exceeds 8 MiB inspection bound')
            (dest/Path(member.name).name).write_bytes(archive.extractfile(member).read())
        sources={Path(m.name).name:m for m in members if m.isfile() and '/csl/' in m.name and m.name.endswith('.csl')}
        for source in root.glob('*.csl'):
            if source.name not in sources or archive.extractfile(sources[source.name]).read()!=source.read_bytes():
                raise ValueError('Embedded source differs from frozen full-model CSL')
    sram=check(root)
    (root/'PRE_RUN_ADMISSION.json').write_text(json.dumps(dict(
      passed=True,full_geometry_compiled=True,full_geometry_simulated=False,
      physical_application_pes=870000,actual_unique_programs=len(apps),
      every_application_elf_sram_passed=sram['passed'],embedded_sources_identical=True,
      allocated_inference_device=False,artifact_sha256=digest(artifact),
      compressed_bytes=artifact.stat().st_size,unpacked_bytes=sum(m.size for m in members),
      recovered_from=recovery['source'] if recovery else None),indent=2)+'\n')
    (root/'artifact.json').write_text(json.dumps(dict(artifact=str(artifact),flags=flags,sha256=digest(artifact)),indent=2)+'\n')
    verify()

if __name__=='__main__':main()
