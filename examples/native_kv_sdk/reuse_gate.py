"""Hash the accepted compiled files and matching CSL; never invoke a compiler."""
import hashlib
import json
from pathlib import Path

COMPILED_SHA256 = '59724c672b010969bef9c794df46202988c57cc57659ea4a6345cd3413c26da0'
PARENT_MANIFEST_SHA256 = '6acaac3a80cde49c1f7ad46319b1f4af3f0aea3b4f5d761b54edd986cfbd935d'


AUXILIARY = json.loads((Path(__file__).resolve().parent / 'runtime-auxiliary.json').read_bytes())['files']


def check_auxiliary(root, base_files, phase):
    if phase not in ('absent', 'live', 'exact'):
        raise ValueError('Known reuse lifetime phase required')
    present = {str(path.relative_to(root)) for path in (root / 'out').rglob('*') if path.is_file()}
    extra = present - set(base_files)
    if not set(base_files) <= present or not extra <= set(AUXILIARY):
        raise ValueError('Missing compiled base or unexpected runtime output')
    if phase == 'absent' and extra:
        raise ValueError('Runtime auxiliary files must be absent before initial load')
    if phase == 'exact' and extra != set(AUXILIARY):
        raise ValueError('All three SDK auxiliary files required after normal stop')
    for name in extra:
        path = root / name
        spec = AUXILIARY[name]
        if path.is_symlink() or not path.is_file() or path.stat().st_size > spec['bytes']:
            raise ValueError('Regular bounded SDK auxiliary: ' + name)
        if phase == 'exact':
            raw = path.read_bytes()
            if len(raw) != spec['bytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
                raise ValueError('Exact SDK auxiliary identity: ' + name)
    return dict(phase=phase, files=sorted(extra))


def verify_reuse(root, auxiliary='live'):
    root = Path(root)
    raw = (root / 'compiled.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != COMPILED_SHA256:
        raise ValueError('Exact accepted compiled receipt')
    compiled = json.loads(raw)
    files = compiled['files']
    if len(files) != 63 or sum(v['bytes'] for v in files.values()) != 2590378 or not compiled['passed']:
        raise ValueError('Exact accepted compiled inventory')
    check_auxiliary(root, files, auxiliary)
    parent_raw = (root / 'parent-manifest.json').read_bytes()
    if hashlib.sha256(parent_raw).hexdigest() != PARENT_MANIFEST_SHA256:
        raise ValueError('Exact parent source manifest')
    parent = json.loads(parent_raw)['files']
    for name, spec in parent.items():
        if not name.endswith('.csl'):
            continue
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Compiled CSL identity: ' + name)
    for name, spec in files.items():
        if not name.startswith('out/') or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Safe compiled path')
        path = root / name
        if path.is_symlink() or path.stat().st_size != spec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
            raise ValueError('Accepted compiled file changed: ' + name)
    return dict(files=63, bytes=2590378, compiled_sha256=COMPILED_SHA256,
                compiler_invoked=False)
