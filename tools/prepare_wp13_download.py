"""Freeze the bounded16MiB selected-weight acquisition; no network request here."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


def prepare(root,run):
    run=run.resolve();run.mkdir(parents=True,exist_ok=False)
    (run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir()
    for name in ('__init__.py','resources.py','locking.py','ranged.py'):
        shutil.copyfile(root/'core/qwen38'/name,run/'core/qwen38'/name)
    for name in ('guarded_run.py','download_wp13_head.py'):
        shutil.copyfile(root/'tools'/name,run/'tools'/name)
    shutil.copyfile(root/'examples/wp13/acquisition-plan.json',run/'plan.json')
    spec={'planned_write_bytes':16*1024*1024,'required_profile':'cpu-reference',
          'steps':[{'name':'download','seconds':60,'argv':[sys.executable,
                    'tools/download_wp13_head.py','--plan','plan.json','--output','weights']}]}
    (run/'steps.json').write_text(json.dumps(spec,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    print(prepare(Path(__file__).resolve().parents[1],args.run))
