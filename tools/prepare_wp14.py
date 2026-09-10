"""Freeze a WP14 integration proposal. This preparer never runs SDK commands."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from qwen38.wp14_layout import budget,sequence,declared_bytes
from qwen38.trained_qk_rope_numerics import POLICY


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference(path):
    manifest=json.loads((path/'source-manifest.json').read_text())
    for name,h in manifest.items():
        if digest(path/name)!=h: raise ValueError('Frozen CPU source changed: '+name)
    result=json.loads((path/'result.json').read_text())
    assert result['passed']
    return result


def prepare(run,image,reference,projection_reference):
    run=run.resolve();reference=reference.resolve();projection_reference=projection_reference.resolve();image=image.resolve()
    assert image.is_file()
    qk_result=verify_reference(reference);proj_result=verify_reference(projection_reference)
    assert digest(reference/'observations.npz')==qk_result['output_sha256']
    assert digest(projection_reference/'observations.npz')==proj_result['observations_sha256']
    qk_plan=json.loads((reference/'reference-plan.json').read_text())
    assert qk_plan['policy']==POLICY and qk_plan['positions']==[1,2,3,0]
    for name,h in qk_plan['frozen_inputs'].items(): assert digest(reference/name)==h
    ref=json.loads((projection_reference/'reference-plan.json').read_text())
    weights=Path(ref['weights_directory'])
    ref['weight_files']={n:h for n,h in ref['weight_files'].items() if n in ('q-with-norm.bf16','k-with-norm.bf16')}
    assert len(ref['weight_files'])==2
    for name,identity in ref['weight_files'].items(): assert digest(weights/name)==identity['sha256']
    run.mkdir(parents=True,exist_ok=False)
    for directory in ('core/qwen38','tools','weights'): (run/directory).mkdir(parents=True)
    for name in ('driver.py','consumer_checks.py','compile_entry.py','producer.csl','consumer.csl','layout.csl'):
        shutil.copyfile(ROOT/'examples/wp14'/name,run/name)
    for source,target in [('persistent_gemv_bf16_f32.csl','kernel.csl'),
                          ('qk_rms256_rope64_bf16.csl','qk_kernel.csl'),('sincos_bounded_f32.csl','trig_kernel.csl')]:
        shutil.copyfile(ROOT/'csl/kernels'/source,run/target)
    for name in ('__init__.py','resources.py','locking.py','elf_inventory.py','rms_numerics.py','gated_numerics.py',
                 'interval_numerics.py','attention_numerics.py','qk_rope_numerics.py','rotary_trig.py',
                 'qk_rope_protocol.py','trained_qk_rope_numerics.py','projection_handoff.py','wp14_layout.py'):
        shutil.copyfile(ROOT/'core/qwen38'/name,run/'core/qwen38'/name)
    for name in ('guarded_run.py','sdk_container.py','prepare_wp14.py'):
        shutil.copyfile(ROOT/'tools'/name,run/'tools'/name)
    for name in ('hidden.bf16','observations.npz'):
        shutil.copyfile(projection_reference/name,run/name)
    for source,target in [('observations.npz','qk-reference-observations.npz'),('result.json','qk-reference-result.json'),
                          ('reference-plan.json','qk-reference-plan.json')]:
        shutil.copyfile(reference/source,run/target)
    shutil.copyfile(projection_reference/'result.json',run/'reference-result.json')
    shutil.copyfile(weights/'download.json',run/'weight-receipt.json')
    for name in ref['weight_files']: shutil.copyfile(weights/name,run/'weights'/name)
    (run/'reference-plan.json').write_text(json.dumps(ref,indent=2)+'\n')
    (run/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    shutil.copyfile(ROOT/'docs/WP14-PROTOCOL.md',run/'PROTOCOL.md')
    resource=json.loads((ROOT/'examples/wp14/resource-plan.json').read_text())
    assert resource['traffic']==budget()
    for role in ('producer','consumer'):
        assert resource['buffers'][role]==json.loads(json.dumps(declared_bytes(role)))
    shutil.copyfile(ROOT/'examples/wp14/resource-plan.json',run/'resource-plan.json')
    (run/'physical-sequence.json').write_text(json.dumps(sequence(),indent=2)+'\n')
    for path in run.rglob('*.py'): ast.parse(path.read_text())
    for path in run.glob('*.csl'):
        source=re.sub(r'//[^\n]*|"(?:\\.|[^"\\])*"','',path.read_text());stack=[]
        for char in source:
            if char in '([{': stack.append(char)
            elif char in ')]}':
                if not stack or stack.pop()!={')':'(',']':'[','}':'{'}[char]: raise ValueError('CSL delimiter mismatch: '+path.name)
        if stack: raise ValueError('Unclosed CSL delimiters: '+path.name)
    profile=dict(scope='wp14-original-hidden-qk-handoff-rms-rope',pes=5,producer_slabs=[0,1,4,5],
                 columns=5120,case_indices=[0],position=1,traffic=budget(),
                 observations_sha256=proj_result['observations_sha256'],qk_observations_sha256=qk_result['output_sha256'],
                 input_sha256={'bounded_dyadic':hashlib.sha256((run/'hidden.bf16').read_bytes()[:10240]).hexdigest()},
                 hidden_file_sha256=digest(run/'hidden.bf16'),
                 dense_vector_sha256=hashlib.sha256((run/'hidden.bf16').read_bytes()[:10240]).hexdigest(),
                 estimate_seconds=265,hard_seconds=300,
                 estimate_basis=resource['estimate_caveat'],
                 sdk_launch_permission='External controller review required. No SDK launch is authorized by freeze creation.',
                 retained_weight_payload_bytes=sum((run/'weights'/n).stat().st_size for n in ref['weight_files']),
                 omitted_weight_payload='Original V file unused by this Q/K-only device chain; retained original Q-with-norm includes unused rawgate rows.',
                 source_boundary='Frozen source intervals from original hidden inputs plus separate actual-producer conditional stages and exact casts/transport/tails.')
    (run/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    steps=dict(required_profile='sdk',planned_write_bytes=134217728,steps=[
        dict(name='compile',seconds=300,argv=prefix+['compile_entry.py','layout.csl','--arch=wse3',
            '--fabric-dims=12,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1',
            '--max-parallelism=1','--dump-dsr-alloc-graph']),
        dict(name='simulate',seconds=300,argv=prefix+['driver.py'])])
    (run/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
    files={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps(dict(files=files,scope=profile['scope']),indent=2)+'\n')
    print(json.dumps(dict(run=str(run),files=len(files),manifest_sha256=digest(run/'source-manifest.json'),sdk_executed=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','image','reference','projection-reference'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    prepare(args.run,args.image,args.reference,args.projection_reference)
