"""Export only the performance subtree; leave the functional baseline unchanged.

Use git to inspect/commit/push separately. This tool neither claims acceptance
nor submits hardware. Execution hashes and publication hashes remain distinct.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SUBSTITUTIONS = [('/path/to/alcf-session.sh', '/path/to/alcf-session.sh'), ('/opt/cerebras/venv/bin/python', '/opt/cerebras/venv/bin/python'), ('/srv/model-storage/qwen38-singlewse', '/srv/model-storage/qwen38-singlewse'), ('/srv/model-storage/', '/srv/model-storage/'), ('/opt/cerebras/sdk/2.10.1', '/opt/cerebras/sdk/2.10.1'), ('/srv/cerebras-workstation/heavy.lock', '/srv/cerebras-workstation/heavy.lock'), ('/srv/qwen38-singlewse-hardware', '/srv/qwen38-singlewse-hardware'), ('/srv/cerebras-hardware/hardware.lock', '/srv/cerebras-hardware/hardware.lock')]


def sha(raw):return hashlib.sha256(raw).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    if not (a.repo/'.git').is_dir():raise ValueError('Publication clone required')
    target=a.repo/'models/qwen38-singlewse/performance';entries={}
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts:continue
        rel=f.relative_to(ROOT)
        if f.name in ['PUBLICATION_RECEIPT.json']:continue
        if f.suffix not in ['.py','.csl','.json','.md']:continue
        if f.name in ['dispatch.json','staging.json']:continue
        raw=f.read_bytes()
        if len(raw)>8<<20:raise ValueError('Noncompact artifact: '+str(rel))
        text=raw.decode()
        if f.suffix!='.csl':
            for before,after in SUBSTITUTIONS:text=text.replace(before,after)
        if f.suffix=='.py':ast.parse(text)
        if f.suffix=='.json':json.loads(text)
        published=text.encode();dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(published)
        entries[str(rel)]=dict(source_sha256=sha(raw),published_sha256=sha(published),site_adapted=raw!=published)
    (target/'SOURCE_EXPORT.json').write_text(json.dumps(dict(files=entries,note='Execution manifests retain original hashes; only non-CSL site paths are adapted. No model weights or compiled artifacts.'),indent=2)+'\n')
    # The repository manifest is a flat map; preserve its existing convention.
    manifest={}
    for f in sorted(a.repo.rglob('*')):
        if not f.is_file() or '.git' in f.relative_to(a.repo).parts or f==a.repo/'PUBLIC_MANIFEST.json':continue
        if '__pycache__' in f.parts or f.suffix=='.pyc':continue
        manifest[str(f.relative_to(a.repo))]=sha(f.read_bytes())
    (a.repo/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(entries),destination=str(target),repository_manifest_files=len(manifest))))

if __name__=='__main__':main()
