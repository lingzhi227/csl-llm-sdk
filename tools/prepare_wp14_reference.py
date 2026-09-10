"""Freeze a bounded WP14 CPU candidate using existing WP13 artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'core'))
from qwen38.trained_qk_rope_numerics import POLICY


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run, previous, interpreter):
    run = run.resolve(); previous = previous.resolve()
    recipe = json.loads((ROOT/'examples/wp13/reference-recipe.json').read_text())
    prior = json.loads((previous/'reference-plan.json').read_text())
    result = json.loads((previous/'result.json').read_text())
    assert result['passed'] and digest(previous/'observations.npz') == result['observations_sha256']
    assert digest(previous/'modeling_qwen3_5.py') == recipe['source_sha256']
    assert digest(previous/'hidden.bf16') == recipe['hidden_sha256']
    assert interpreter.is_file()
    weights = Path(prior['weights_directory'])
    for name, identity in prior['weight_files'].items():
        assert digest(weights/name) == identity['sha256']
    run.mkdir(parents=True, exist_ok=False)
    (run/'core/qwen38').mkdir(parents=True); (run/'tools').mkdir()
    for name in ('__init__.py', 'resources.py', 'locking.py', 'rms_numerics.py', 'gated_numerics.py',
                 'interval_numerics.py', 'rotary_trig.py', 'trained_qk_rope_numerics.py'):
        shutil.copyfile(ROOT/'core/qwen38'/name, run/'core/qwen38'/name)
    shutil.copyfile(ROOT/'tools/guarded_run.py', run/'tools/guarded_run.py')
    shutil.copyfile(ROOT/'examples/wp14/build_reference.py', run/'build_reference.py')
    for name in ('modeling_qwen3_5.py', 'hidden.bf16'):
        shutil.copyfile(previous/name, run/name)
    shutil.copyfile(previous/'observations.npz', run/'projection-observations.npz')
    frozen = {name: digest(run/name) for name in ('modeling_qwen3_5.py', 'hidden.bf16', 'projection-observations.npz')}
    frozen.update({str(weights/name): identity['sha256'] for name, identity in prior['weight_files'].items()})
    plan = dict(cases=recipe['cases'], positions=[1, 2, 3, 0], policy=POLICY, weights_directory=str(weights),
                frozen_inputs=frozen, candidate=1, maximum_candidates=2,
                model_revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
                upstream_revision='4815a0a6a064214f2d8208c094464a5a6b76ca8d',
                workload='Four selected Q/rawgate512 and K256 projections, full5120 columns:15728640 products; four Q/K RMS256 and RoPE64 references, source intervals before observations.',
                limits='Existing CPU interpreter only; one thread,2GiB RAM,Swap0,60seconds; no SDK/GPU/download/install; output budget8MiB.')
    (run/'reference-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    steps = dict(required_profile='cpu-reference', planned_write_bytes=8*1024*1024,
                 steps=[dict(name='reference', seconds=60, argv=[str(interpreter.absolute()), 'build_reference.py'])])
    (run/'steps.json').write_text(json.dumps(steps, indent=2)+'\n')
    manifest = {str(p.relative_to(run)): digest(p) for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(run=str(run), files=len(manifest), manifest_sha256=digest(run/'source-manifest.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.run, args.previous, args.python)
