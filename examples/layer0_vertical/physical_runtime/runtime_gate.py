"""Fail-closed physical/offline admission and immutable external input pins."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def pinned(path,pin,limit):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=pin['bytes'] or pin['bytes']>limit:
        raise ValueError('Exact bounded external input '+str(path))
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('External bytes changed '+str(path))
    return raw

def verify(kind):
    if kind not in ('runtime','audit'):raise ValueError('Known exact owner')
    manifest_raw=(ROOT/'runtime-manifest.json').read_bytes();manifest=json.loads(manifest_raw)
    for name,pin in manifest['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Frozen relative source')
        pinned(ROOT/name,pin,4<<20)
    profile=json.loads((ROOT/'runtime-profile.json').read_bytes())
    admission=json.loads((ROOT/('runtime-admission.json' if kind=='runtime' else 'audit-admission.json')).read_bytes())
    expected=dict(candidate=ROOT.name,manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                  kind=kind,profile=profile[kind],automatic_retry=False)
    if admission!=expected:raise ValueError('Exact separate controller '+kind+' admission required')
    inputs=json.loads((ROOT/'runtime-inputs.json').read_bytes())
    if ROOT.stat().st_dev!=51:raise ValueError('Physical candidate and evidence stay on ALCF')
    bindings=json.loads(pinned(inputs['bindings_path'],inputs['bindings_pin'],8<<20))
    if bindings['application']!=[30,1160] or bindings['inputs']['compiled']!=inputs['compiled_pin']:
        raise ValueError('Actual revised vertical compile and bound application identity')
    if Path(inputs['compiled_path']).parent.name!='layer0-vertical-hw-compile-002':
        raise ValueError('New top-layout application inspection is required')
    pinned(inputs['compiled_path'],inputs['compiled_pin'],8<<20)
    if Path(inputs['prepared_root']).name!='layer0-vertical-prepared-002':
        raise ValueError('Top-layout preparation; unused bottom candidate cannot substitute')
    pinned(Path(inputs['prepared_root'])/'prepared.json',inputs['prepared_pin'],2<<20)
    for name,key in [('host-map.json','host_map'),('EXPORTS.json','exports')]:
        pinned(ROOT/name,bindings['inputs'][key],2<<20)
    if not bindings['all_planned_transfers_within_actual_arrays']:raise ValueError('Actual export/bank extent proof')
    return inputs,profile,bindings

def artifact(inputs):
    # runtime-inputs is part of the immutable, separately admitted manifest.
    # Its exact real artifact/report pins are supplied only after acceptance.
    pin=inputs['artifact_pin']
    if Path(inputs['artifact_path']).parent.name!='layer0-vertical-hw-compile-002':
        raise ValueError('Accepted revised vertical artifact is required')
    pinned(inputs['artifact_path'],pin,32<<20)
    return Path(inputs['artifact_path'])
