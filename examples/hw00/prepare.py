"""Prepare a fresh physical reproduction; never submits a hardware job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared',type=Path,required=True,help='Private accepted resident source preparation directory')
    parser.add_argument('--work',type=Path,required=True,help='New empty execution directory on the appliance host')
    args=parser.parse_args()
    source=Path(__file__).resolve().parent
    root=source.parents[1]
    args.work.mkdir(parents=True,exist_ok=False)
    for path in source.iterdir():
        if path.suffix in ('.py','.json','.csl') and path.name not in ('prepare.py','test_hw00.py'):
            shutil.copy2(path,args.work/path.name)
    shutil.copytree(source/'hw00',args.work/'hw00',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root/'core',args.work/'core',ignore=shutil.ignore_patterns('__pycache__'))
    names=('preparation.json','hidden.bf16','oracle.npz','weights/gate_proj.bf16','weights/up_proj.bf16','weights/down_proj.bf16')
    for name in names:
        path=args.work/'prepared'/name;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.prepared/name,path)
    files={}
    for path in sorted(args.work.rglob('*')):
        if path.is_file():
            raw=path.read_bytes();files[str(path.relative_to(args.work))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    raw=(json.dumps(dict(candidate='hw00-b-001',files=files),sort_keys=True,indent=2)+'\n').encode()
    (args.work/'manifest.json').write_bytes(raw)
    template=dict(candidate='hw00-b-001',manifest_sha256=hashlib.sha256(raw).hexdigest(),
                  physical_jobs=1,compile_timeout_s=600,run_timeout_s=300,
                  note='Review storage, SDK path, credentials and resource ownership; rename to admission.json only when ready to allow one physical job.')
    (args.work/'admission.template.json').write_text(json.dumps(template,indent=2)+'\n')
    print('Prepared only:',args.work,'manifest',template['manifest_sha256'])


if __name__=='__main__':main()
