"""Freeze the WP15 two-token10PE proposal; never invokes SDK or supervisor."""
import argparse,ast,hashlib,json,re,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from qwen38.wp15_layout import budget,sequence,resource_plan
from qwen38.projected_attention_numerics import POLICY
from audit_wp15_host_schedule import audit


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference(path):
    manifest=json.loads((path/'source-manifest.json').read_text())
    for name,h in manifest.items():assert digest(path/name)==h,name
    result=json.loads((path/'result.json').read_text());assert result['passed']
    return manifest,result


def prepare(run,image,reference,projection_reference):
    run=run.resolve();reference=reference.resolve();previous=projection_reference.resolve();image=image.resolve()
    assert image.is_file()
    cpu_manifest,cpu=verify_reference(reference);_,projection=verify_reference(previous)
    assert digest(reference/'observations.npz')==cpu['observations_sha256']
    assert digest(reference/'source-intervals.npz')==cpu['source_intervals_sha256']
    assert digest(previous/'observations.npz')==projection['observations_sha256']
    cpu_plan=json.loads((reference/'reference-plan.json').read_text());assert cpu_plan['policy']==POLICY
    for name,h in cpu_plan['frozen_inputs'].items():assert digest(reference/name)==h,name
    assert digest(ROOT/'core/qwen38/projected_attention_numerics.py')==cpu_manifest['core/qwen38/projected_attention_numerics.py']
    ref=json.loads((previous/'reference-plan.json').read_text());weights=Path(ref['weights_directory'])
    for name,h in ref['weight_files'].items():assert digest(weights/name)==h['sha256']
    assert len(ref['weight_files'])==3
    run.mkdir(parents=True,exist_ok=False)
    for d in ('core/qwen38','tools','weights','reference-cpu'):(run/d).mkdir(parents=True)
    for name in ('driver.py','consumer_checks.py','compile_entry.py','producer.csl','qk.csl','attention.csl','layout.csl'):
        shutil.copyfile(ROOT/'examples/wp15'/name,run/name)
    for src,dst in [('persistent_gemv_bf16_f32.csl','kernel.csl'),('qk_rms256_rope64_bf16.csl','qk_kernel.csl'),
                    ('sincos_bounded_f32.csl','trig_kernel.csl'),('attention256_bf16.csl','attention_kernel.csl'),('exp_bounded_f32.csl','exp_kernel.csl')]:
        shutil.copyfile(ROOT/'csl/kernels'/src,run/dst)
    for name in ('__init__','resources','locking','elf_inventory','rms_numerics','gated_numerics','interval_numerics',
                 'attention_numerics','qk_rope_numerics','rotary_trig','qk_rope_protocol','trained_qk_rope_numerics',
                 'wp14_layout','attention_protocol','attention_composed_numerics','projected_attention_numerics',
                 'projected_attention_protocol','projected_attention_checks','wp15_layout','wp15_admission'):
        shutil.copyfile(ROOT/'core/qwen38'/(name+'.py'),run/'core/qwen38'/(name+'.py'))
    for name in ('guarded_run.py','guarded_wp15.py','sdk_container.py','prepare_wp15.py','audit_wp15_host_schedule.py'):
        shutil.copyfile(ROOT/'tools'/name,run/'tools'/name)
    # The CPU recipe/source and original small input/oracle files are immutable
    # context, not invoked from this SDK recipe. Their old absolute weight paths
    # do not replace the run-local payload used by driver.py.
    for name in (*cpu_manifest,'source-manifest.json','result.json','supervisor.json','completion-audit.json','conditional-sensitivity.json'):
        target=run/'reference-cpu'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(reference/name,target)
    for source,target in [('observations.npz','attention-reference-observations.npz'),('source-intervals.npz','source-intervals.npz')]:
        shutil.copyfile(reference/source,run/target)
    for name in ('hidden.bf16','observations.npz'):shutil.copyfile(previous/name,run/name)
    shutil.copyfile(previous/'result.json',run/'reference-result.json')
    shutil.copyfile(weights/'download.json',run/'weight-receipt.json')
    for name in ref['weight_files']:shutil.copyfile(weights/name,run/'weights'/name)
    (run/'reference-plan.json').write_text(json.dumps(ref,indent=2)+'\n')
    (run/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    for source,target in [('WP15-PROTOCOL.md','PROTOCOL.md'),('WP15-REPORT.md','CPU-REPORT.md')]:
        shutil.copyfile(ROOT/'docs'/source,run/target)
    resource=resource_plan()
    assert json.loads((ROOT/'examples/wp15/resource-plan.json').read_text())==json.loads(json.dumps(resource))
    (run/'resource-plan.json').write_text(json.dumps(resource,indent=2)+'\n')
    (run/'physical-sequence.json').write_text(json.dumps(sequence(),indent=2)+'\n')
    profile=dict(scope='wp15-original-hidden-persistent-attention',pes=10,producer_slabs=list(range(8)),columns=5120,
                 case_indices=[0,1],positions=[0,1],request_generations=[1,1],contraction_ids=[1,2],traffic=budget(),
                 observations_sha256=projection['observations_sha256'],attention_observations_sha256=cpu['observations_sha256'],
                 source_intervals_sha256=cpu['source_intervals_sha256'],cpu_source_manifest_sha256=digest(reference/'source-manifest.json'),
                 input_sha256={ref['cases'][i]:hashlib.sha256((run/'hidden.bf16').read_bytes()[i*10240:(i+1)*10240]).hexdigest() for i in (0,1)},
                 hidden_file_sha256=digest(run/'hidden.bf16'),estimate_seconds=1100,hard_seconds=1200,compile_hard_seconds=300,
                 estimate_basis=resource['estimate_basis'],sdk_execution_authorized=False,
                 sdk_launch_permission='Exact controller manifest admission required by guarded_wp15.py. Freeze creation does not admit SDK work.',
                 retained_weight_payload_bytes=sum((run/'weights'/n).stat().st_size for n in ref['weight_files']),
                 conditional_ignore_QK_witness_required_on_token2=True,
                 source_boundary='Frozen source intervals preserve original projection and historical cache uncertainty; actual-operand conditional checks remain separate.')
    (run/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    steps=dict(required_profile='sdk',planned_write_bytes=134217728,steps=[
        dict(name='compile',seconds=300,argv=prefix+['compile_entry.py','layout.csl','--arch=wse3','--fabric-dims=17,3',
            '--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']),
        dict(name='simulate',seconds=1200,argv=prefix+['driver.py'])])
    (run/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
    for path in run.rglob('*.py'):ast.parse(path.read_text())
    for path in run.glob('*.csl'):
        text=re.sub(r'//[^\n]*|"(?:\\.|[^"\\])*"','',path.read_text());stack=[]
        for char in text:
            if char in '([{':stack.append(char)
            elif char in ')]}':assert stack and stack.pop()=={')':'(',']':'[','}':'{'}[char],path.name
        assert not stack,path.name
    (run/'host-static-audit.json').write_text(json.dumps(audit(run),indent=2)+'\n')
    files={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps(dict(files=files,scope=profile['scope']),indent=2)+'\n')
    print(json.dumps(dict(run=str(run),files=len(files),bytes=sum((run/n).stat().st_size for n in files),
                         manifest_sha256=digest(run/'source-manifest.json'),sdk_executed=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','image','reference','projection-reference'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();prepare(args.run,args.image,args.reference,args.projection_reference)
