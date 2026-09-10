"""Freeze one original-hidden persistent-attention CPU candidate; no launch."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from qwen38.projected_attention_numerics import POLICY


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run,projection_reference,qk_reference,interpreter):
    run=run.resolve();previous=projection_reference.resolve();qk_reference=qk_reference.resolve()
    p=json.loads((previous/'reference-plan.json').read_text());result=json.loads((previous/'result.json').read_text())
    assert result['passed'] and digest(previous/'observations.npz')==result['observations_sha256']
    for name,h in json.loads((previous/'source-manifest.json').read_text()).items():assert digest(previous/name)==h
    qr=json.loads((qk_reference/'result.json').read_text())
    assert qr['passed'] and digest(qk_reference/'observations.npz')==qr['output_sha256']
    for name,h in json.loads((qk_reference/'source-manifest.json').read_text()).items():assert digest(qk_reference/name)==h
    weights=Path(p['weights_directory'])
    for name,h in p['weight_files'].items():assert digest(weights/name)==h['sha256']
    assert interpreter.is_file()
    run.mkdir(parents=True,exist_ok=False);(run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir()
    for name in ('__init__.py','resources.py','locking.py','rms_numerics.py','gated_numerics.py','interval_numerics.py',
                 'rotary_trig.py','trained_qk_rope_numerics.py','projected_attention_numerics.py'):
        shutil.copyfile(ROOT/'core/qwen38'/name,run/'core/qwen38'/name)
    for name in ('build_reference.py','pinned_reference.py'):
        shutil.copyfile(ROOT/'examples/wp15'/name,run/name)
    shutil.copyfile(ROOT/'tools/guarded_run.py',run/'tools/guarded_run.py')
    for source,target in [('modeling_qwen3_5.py','modeling_qwen3_5.py'),('hidden.bf16','hidden.bf16'),('observations.npz','projection-observations.npz')]:
        shutil.copyfile(previous/source,run/target)
    frozen={n:digest(run/n) for n in ('modeling_qwen3_5.py','hidden.bf16','projection-observations.npz')}
    frozen.update({str(weights/n):h['sha256'] for n,h in p['weight_files'].items()})
    plan=dict(candidate=1,maximum_candidates=2,policy=POLICY,weights_directory=str(weights),frozen_inputs=frozen,
              inverse_frequency_bits=qr['inverse_frequency_bits'],
              model_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',upstream_revision='4815a0a6a064214f2d8208c094464a5a6b76ca8d',
              source_archive_before_observation=True,
              workload='4*1024*5120=20971520 selected CPU projection products; trained Q/K/RoPE and persistent attention references with valid lengths1,2,3,1.25 small eager attention calls including predeclared counterexamples; two extra rotary calls reuse saved normalized vectors, no extra GEMV.',
              counterexamples=['zero','cyclic_dimension_permutation','no_sigmoid_gate','current_KV_only','reset_cache_each_token',
                  'ignore_QK_uniform_scores','negate_current_query','wrong_V_routed_from_rawgate','reverse_aligned_cache_pairs','reverse_two_hidden_inputs_and_positions'],
              counterexample_scope='Aligned K/V order alone is attention permutation-invariant and needs exact cache/protocol checks; reversed original input sequence changes current operands and positions.',
              limits='Existing CPU interpreter only,one thread,2GiB RAM,Swap0,60seconds,8MiB planned writes; no SDK/GPU/modeldownload/environment installation.')
    (run/'reference-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    steps=dict(required_profile='cpu-reference',planned_write_bytes=8*1024*1024,
               steps=[dict(name='reference',seconds=60,argv=[str(interpreter.absolute()),'build_reference.py'])])
    (run/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
    for path in run.rglob('*.py'):ast.parse(path.read_text())
    manifest={str(p.relative_to(run)):digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),files=len(manifest),manifest_sha256=digest(run/'source-manifest.json'),executed=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','projection-reference','qk-reference','python'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();prepare(args.run,args.projection_reference,args.qk_reference,args.python)
