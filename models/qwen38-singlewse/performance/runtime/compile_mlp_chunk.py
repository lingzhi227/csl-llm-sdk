"""Compile a bounded component and admit every loaded PE, including shared images."""
import hashlib
import json
import logging
from pathlib import Path
import tarfile
from job_capture import install
from source_gate import verify
from check_sram import check


def main():
    verify(); install('compile.jobs'); logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkCompiler
    root = Path.cwd(); config = json.loads((root / 'experiment.json').read_text())
    if config['application'] != [3, 3] or config['application_pes'] != 9:
        raise ValueError('Qualified bounded component geometry required')
    flags = '--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out'
    with SdkCompiler(disable_version_check=True, resource_cpu=2000, resource_mem=4 << 30) as compiler:
        artifact = Path(compiler.compile(str(root), 'layout.csl', flags, str(root)))
    if artifact.stat().st_size > 8 << 20:
        raise ValueError('Compressed artifact budget')
    with tarfile.open(artifact) as archive:
        members = archive.getmembers()
        if len(members) > 512 or sum(m.size for m in members) > 8 << 20:
            raise ValueError('Expanded artifact budget')
        apps = [m for m in members if m.isfile() and '/out/bin/' in m.name and m.name.endswith('.elf')]
        if not 1 <= len(apps) <= config['application_pes']:
            raise ValueError('Application image count budget')
        destination = root / 'out/bin'; destination.mkdir(parents=True)
        names = set()
        for member in apps:
            name = Path(member.name).name
            if name in names:
                raise ValueError('Duplicate image member')
            names.add(name); (destination / name).write_bytes(archive.extractfile(member).read())
        sources = {Path(m.name).name: m for m in members if m.isfile() and '/csl/' in m.name and m.name.endswith('.csl')}
        for source in root.glob('*.csl'):
            if source.name not in sources or archive.extractfile(sources[source.name]).read() != source.read_bytes():
                raise ValueError('Embedded CSL identity mismatch: ' + source.name)
    result = check(root)
    if result['application_pes'] != config['application_pes']:
        raise ValueError('Coordinate coverage mismatch')
    (root / 'artifact.json').write_text(json.dumps(dict(artifact=str(artifact), flags=flags,
                                                      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()), indent=2) + '\n')
    verify()


if __name__ == '__main__':
    main()
