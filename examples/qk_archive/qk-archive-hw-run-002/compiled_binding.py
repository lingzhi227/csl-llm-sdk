"""Bind a separately accepted physical compile artifact before device allocation."""
import hashlib,json
from pathlib import Path
from source_gate import ROOT


def preflight():
    binding=json.loads((ROOT/'compiled-binding.json').read_bytes())
    parent=ROOT.parent/'qk-archive-hw-compile-001'
    if parent.is_symlink() or parent.stat().st_dev!=ROOT.stat().st_dev:
        raise ValueError('Exact native compile candidate on the same filesystem')
    if set(binding['receipts'])!={'manifest.json','artifact.json','sram.json','placement.json','COMPLETE.json','compile-audit.json'}:
        raise ValueError('Complete accepted compile receipt set')
    for name,spec in binding['receipts'].items():
        path=parent/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or path.stat().st_size!=spec['bytes'] or spec['bytes']>1048576:
            raise ValueError('Bounded compile receipt')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError('Accepted compile receipt identity')
    if not all(json.loads((parent/name).read_bytes())['passed'] for name in ('sram.json','placement.json')):
        raise ValueError('Actual physical compile gates')
    placement=json.loads((parent/'placement.json').read_bytes())
    if placement['PEs']!=4 or placement['geometry']!=[762,1172]:
        raise ValueError('Exact accepted physical four-PE geometry')
    complete=json.loads((parent/'COMPLETE.json').read_bytes())
    if not complete['all_owned_jobs_released'] or complete['runtime_created']:
        raise ValueError('Accepted compile released without runtime')
    receipt=json.loads((parent/'artifact.json').read_bytes())
    if Path(receipt['artifact']).name!=binding['artifact']['filename'] or receipt['sha256']!=binding['artifact']['sha256']:
        raise ValueError('Accepted artifact receipt binding')
    artifact=parent/binding['artifact']['filename']
    if artifact.parent!=parent or artifact.is_symlink() or artifact.stat().st_size!=binding['artifact']['bytes'] or artifact.stat().st_size>32<<20:
        raise ValueError('Bounded exact compiled artifact')
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=binding['artifact']['sha256']:
        raise ValueError('Compiled artifact bytes changed')
    return artifact,binding
