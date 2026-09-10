"""Freeze the full-head recurrent CPU source-body fixtures; no install, download or execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SOURCES = {
    'modeling_qwen3_5.py': '762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b',
}


def prepare(root, run, sources, interpreter):
    run = run.resolve()
    interpreter = interpreter.absolute()  # Preserve a venv entry rather than resolving its base symlink.
    if not interpreter.is_file():
        raise ValueError('Missing prequalified Python interpreter')
    for name, digest in SOURCES.items():
        if hashlib.sha256((sources/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pinned source mismatch: '+name)
    run.mkdir(parents=True, exist_ok=False)
    (run/'core/qwen38').mkdir(parents=True)
    (run/'tools').mkdir()
    for name in SOURCES:
        shutil.copyfile(sources/name, run/name)
    for name in ('__init__.py', 'resources.py', 'locking.py', 'recurrent_numerics.py'):
        shutil.copyfile(root/'core/qwen38'/name, run/'core/qwen38'/name)
    shutil.copyfile(root/'tools/guarded_run.py', run/'tools/guarded_run.py')
    shutil.copyfile(root/'examples/wp06/build_reference.py', run/'build_reference.py')
    spec = {'planned_write_bytes': 10*1024*1024, 'required_profile': 'cpu-reference',
            'steps': [{'name': 'reference', 'seconds': 60, 'argv': [str(interpreter), 'build_reference.py']}]}
    (run/'steps.json').write_text(json.dumps(spec, indent=2)+'\n')
    files = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps(files, indent=2)+'\n')
    return run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(Path(__file__).resolve().parents[1], args.run, args.source_dir, args.python))
