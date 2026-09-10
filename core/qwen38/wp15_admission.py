"""Exact WP15-only controller admission for the longer simulation proposal."""
import hashlib
import json
from pathlib import Path


def validate(admission,profile,spec,manifest_sha256,candidate):
    if admission.get('package')!='WP15' or admission.get('sdk_execution_authorized') is not True:
        raise ValueError('Explicit WP15 controller SDK admission required')
    if admission.get('candidate')!=candidate or admission.get('candidate_limit')!=1 or admission.get('manifest_sha256')!=manifest_sha256:
        raise ValueError('Controller admission does not identify this exact single candidate')
    if profile.get('scope')!='wp15-original-hidden-persistent-attention' or profile.get('pes')!=10:
        raise ValueError('Long deadline is limited to the selected WP15 10PE scope')
    if profile.get('case_indices')!=[0,1] or profile.get('positions')!=[0,1] or profile.get('request_generations')!=[1,1]:
        raise ValueError('Admission requires the frozen two-token same-request scope')
    if spec.get('required_profile')!='sdk' or len(spec.get('steps',[]))!=2:
        raise ValueError('Exactly one compile then one simulate required')
    for step,name,key,cap in zip(spec['steps'],('compile','simulate'),('compile_seconds','simulation_seconds'),(300,1200)):
        limit=admission.get(key)
        if type(limit) is not int or not 1<=limit<=cap or step.get('name')!=name or step.get('seconds')!=limit:
            raise ValueError('Step deadline differs from scoped controller admission')
    return True


def verify(work,spec,admission_file):
    manifest_path=work/'source-manifest.json'
    digest=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest=json.loads(manifest_path.read_text())['files']
    for name,expected in manifest.items():
        path=work/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.is_file():
            raise ValueError('Unsafe frozen source path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen WP15 file changed: '+name)
    return validate(json.loads(admission_file.read_text()),json.loads((work/'profile.json').read_text()),spec,digest,work.name)
