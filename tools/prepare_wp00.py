"""Create a small, new, reproducible WP00 bundle without launching the SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def prepare(root, run, image, python):
    run = Path(run).resolve()
    image = Path(image).resolve()
    if not image.is_file():
        raise ValueError('SDK image must already exist; no automatic downloads')
    run.mkdir(parents=True, exist_ok=False)
    for name in ('layout.csl','pe.csl','driver.py','compile_entry.py'):
        shutil.copyfile(root/'examples/wp00'/name, run/name)
    for name in ('sdk_container.py','guarded_run.py'):
        (run/'tools').mkdir(exist_ok=True)
        shutil.copyfile(root/'tools'/name, run/'tools'/name)
    (run/'core/qwen38').mkdir(parents=True)
    for path in (root/'core/qwen38').glob('*.py'):
        shutil.copyfile(path, run/'core/qwen38'/path.name)
    prefix=[str(python),str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    spec={'planned_write_bytes':134217728,'steps':[
        {'name':'compile','seconds':300,'argv':prefix+['compile_entry.py','layout.csl','--arch=wse3',
            '--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1']},
        {'name':'simulate','seconds':300,'argv':prefix+['driver.py']}]}
    (run/'steps.json').write_text(json.dumps(spec,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    args=parser.parse_args()
    run=prepare(Path(__file__).resolve().parents[1],args.run,args.image,sys.executable)
    print('Prepared only; review manifest and run spec before execution:',run)


if __name__=='__main__':
    main()
