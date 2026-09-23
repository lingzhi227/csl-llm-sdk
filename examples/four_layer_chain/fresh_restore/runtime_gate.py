"""Exact source/external pins and distinct admissions for each physical owner."""
from pathlib import Path
import hashlib
import json
from runtime_boundary import require
ROOT = Path(__file__).resolve().parent


def pinned(path, pin, limit):
    path = Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size == pin['bytes'] and
            0 <= pin['bytes'] <= limit, 'Bounded exact immutable input: '+str(path))
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256'] and
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Pinned input changed: '+str(path))
    return raw


def source():
    raw = (ROOT/'runtime-manifest.json').read_bytes()
    manifest = json.loads(raw)
    for name, pin in manifest['files'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Frozen relative source path')
        require(not (ROOT/name).stat().st_mode & 0o222, 'Frozen source is readonly')
        pinned(ROOT/name, pin, 16 << 20)
    return hashlib.sha256(raw).hexdigest()


def restore_source(inputs):
    """Bind the actual baseline and its separately completed resumed audit."""
    from runtime_restore_provenance import restore_source as restored
    return restored(ROOT, inputs, pinned, require)


def verify(kind):
    require(kind in ('runtime', 'audit'), 'Exact known owner phase')
    manifest_sha256 = source()
    inputs = json.loads((ROOT/'runtime-inputs.json').read_bytes())
    profiles = json.loads((ROOT/'runtime-profile.json').read_bytes())
    require(ROOT.stat().st_dev == 51 and inputs['mode'] in ('baseline', 'restore'), 'ALCF-owned original chain context')
    restore_pin = None if inputs['mode'] == 'baseline' else restore_source(inputs)[0]
    expected = dict(candidate=ROOT.name, manifest_sha256=manifest_sha256, kind=kind,
                    profile=profiles[kind], automatic_retry=False, restore_input_pin=restore_pin)
    admission = json.loads((ROOT/(kind+'-admission.json')).read_bytes())
    require(admission == expected, 'Exact separate admitted source, mode, limits and restore input')
    bindings = json.loads(pinned(inputs['bindings_path'], inputs['bindings_pin'], 16 << 20))
    require(bindings['application'] == [119, 1160] and bindings['inputs']['compiled'] == inputs['compiled_pin'] and
            bindings['all_planned_transfers_within_actual_arrays'], 'Complete actual compiled array binding')
    pinned(ROOT/'host-plan.json', bindings['inputs']['host_plan'], 16 << 20)
    require(Path(inputs['compiled_path']).parent.name == 'layer-chain-hw-compile-002', 'Accepted full actual compilation')
    pinned(inputs['compiled_path'], inputs['compiled_pin'], 32 << 20)
    require(set(inputs['prepared']) == {'0', '1', '2', '3'}, 'Four own original preparations')
    expected_roots = ['layer0-vertical-prepared-002', 'layer-chain-original1-prepared-001',
                      'layer-chain-original2-prepared-001', 'layer3-vertical-prepared-002']
    for layer, item in inputs['prepared'].items():
        require(Path(item['root']).name == expected_roots[int(layer)], 'Accepted original-layer parameter directory')
        pinned(Path(item['root'])/'prepared.json', item['pin'], 2 << 20)
    return inputs, profiles, bindings


def artifact(inputs):
    require(Path(inputs['artifact_path']).parent.name == 'layer-chain-hw-compile-002', 'Accepted full four-layer artifact')
    pinned(inputs['artifact_path'], inputs['artifact_pin'], 128 << 20)
    return Path(inputs['artifact_path'])
