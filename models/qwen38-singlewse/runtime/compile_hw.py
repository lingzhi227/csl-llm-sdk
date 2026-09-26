"""Compile the original-weight tile for the full physical WSE-3 fabric."""
import hashlib
import json
import logging
from pathlib import Path
import tarfile
from job_capture import install
from source_gate import verify
from check_sram import check


def main():
    verify()
    install("compile.jobs")
    logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkCompiler
    root = Path.cwd()
    config = json.loads((root/"experiment.json").read_text()) if (root/"experiment.json").exists() else {}
    resident_roles=config.get('resident_roles',False)
    if resident_roles and config.get('application')!=[12,12]:
        raise ValueError('Unexpected representative role geometry')
    full_expert=config.get('full_expert',False)
    full_matrix=config.get('full_matrix',False)
    if full_matrix and (config.get('application_pes')!=2600 or config.get('matrix_shape')!=[17408,5120]):
        raise ValueError('Unexpected full matrix profile')
    if full_expert and (config.get('application_pes')!=828 or config.get('compiler_params')!='gate_rows:36,down_rows:18'):
        raise ValueError('Full expert geometry differs from the bounded 828-PE profile')
    member_limit=16384 if full_matrix else (4096 if full_expert else 512)
    byte_limit=(512 if full_matrix else (128 if full_expert else 64))<<20
    flags = "--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out"
    if full_expert:flags+=' --params=gate_rows:36,down_rows:18'
    with SdkCompiler(disable_version_check=True, resource_cpu=2000, resource_mem=4<<30) as compiler:
        artifact = Path(compiler.compile(str(root), "layout.csl", flags, str(root)))
    if artifact.stat().st_size > byte_limit:
        raise ValueError("Microkernel artifact size limit")
    if resident_roles:
        from artifact_stream import audit
        result=audit(artifact,root,144,application=(12,12),compressed_limit=64<<20,expanded_limit=64<<20,member_limit=512)
        (root/'sram.json').write_text(json.dumps(result,indent=2)+'\n')
        (root/'artifact.json').write_text(json.dumps(dict(artifact=str(artifact),flags=flags,sha256=result['artifact_sha256']),indent=2)+'\n')
        verify();return
    with tarfile.open(artifact) as archive:
        members = archive.getmembers()
        if len(members) > member_limit or sum(m.size for m in members) > byte_limit:
            raise ValueError("Microkernel archive size limit")
        apps = [m for m in members if m.isfile() and "/out/bin/" in m.name and m.name.endswith(".elf")]
        if len(apps) != config.get("application_pes",1):
            raise ValueError("Application PE count mismatch")
        dest = root / "out/bin"
        dest.mkdir(parents=True)
        for member in apps:
            (dest / Path(member.name).name).write_bytes(archive.extractfile(member).read())
        sources = {Path(m.name).name: m for m in members if m.isfile() and "/csl/" in m.name and m.name.endswith(".csl")}
        for source in root.glob("*.csl"):
            if source.name not in sources or archive.extractfile(sources[source.name]).read() != source.read_bytes():
                raise ValueError("Embedded CSL source identity mismatch")
    check(root)
    (root / "artifact.json").write_text(json.dumps(dict(artifact=str(artifact), flags=flags,
         sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()), indent=2)+"\n")
    verify()


if __name__ == "__main__":
    main()
