"""Export performance source and the two explicitly authorized active notices.

Use git to inspect/commit/push separately. This tool neither claims acceptance
nor submits hardware. Historical inference source/results remain unchanged.
Execution hashes and publication hashes remain distinct.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SUBSTITUTIONS = [(str(ROOT.parent), "/path/to/qwen38-singlewse"), ('/path/to/alcf-session.sh', '/path/to/alcf-session.sh'), ('/opt/cerebras/venv/bin/python', '/opt/cerebras/venv/bin/python'), ('/srv/model-storage/qwen38-singlewse', '/srv/model-storage/qwen38-singlewse'), ('/srv/model-storage/', '/srv/model-storage/'), ('/opt/cerebras/sdk/2.10.1', '/opt/cerebras/sdk/2.10.1'), ('/srv/cerebras-workstation/heavy.lock', '/srv/cerebras-workstation/heavy.lock'), ('/srv/qwen38-singlewse-hardware', '/srv/qwen38-singlewse-hardware'), ('/srv/cerebras-hardware/hardware.lock', '/srv/cerebras-hardware/hardware.lock')]


def sha(raw):return hashlib.sha256(raw).hexdigest()


def validate_python(path, raw, published):
    """Preserve a proven failed execution source, without accepting new errors."""
    try:
        ast.parse(published)
    except SyntaxError as error:
        rel=path.relative_to(ROOT)
        if len(rel.parts)<4 or rel.parts[0]!='evidence' or rel.parts[2]!='source':raise
        receipt=ROOT/rel.parts[0]/rel.parts[1]/'source-syntax-errors.json'
        if not receipt.is_file():raise
        record=json.loads(receipt.read_text()).get('/'.join(rel.parts[3:]))
        if not record or record['source_sha256']!=sha(raw):raise
        # An adaptation that newly corrupts otherwise valid Python is never
        # admitted. The original, observed failure must have the same location.
        try:ast.parse(raw)
        except SyntaxError as original:
            if original.lineno!=record['line'] or error.lineno!=original.lineno or original.msg!=record['message']:raise
        else:raise ValueError('Publication introduced a new Python syntax error')

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
    if not (a.repo/'.git').is_dir():raise ValueError('Publication clone required')
    target=a.repo/'models/qwen38-singlewse/performance';entries={}
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts:continue
        rel=f.relative_to(ROOT)
        if f.name in ['PUBLICATION_RECEIPT.json']:continue
        if f.suffix not in ['.py','.csl','.cslpart','.json','.md']:continue
        if f.name in ['dispatch.json','staging.json']:continue
        raw=f.read_bytes()
        if len(raw)>8<<20:raise ValueError('Noncompact artifact: '+str(rel))
        text=raw.decode()
        if f.suffix not in ['.csl','.cslpart']:
            for before,after in SUBSTITUTIONS:text=text.replace(before,after)
        if f.suffix=='.py':validate_python(f,raw,text)
        if f.suffix=='.json':json.loads(text)
        published=text.encode();dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(published)
        entries[str(rel)]=dict(source_sha256=sha(raw),published_sha256=sha(published),site_adapted=raw!=published)
    (target/'SOURCE_EXPORT.json').write_text(json.dumps(dict(files=entries,note='Execution manifests retain original hashes; only non-CSL site paths are adapted. No model weights or compiled artifacts.'),indent=2)+'\n')
    notices={}
    for rel in ['docs/STATUS.md','docs/ACCEPTANCE.md']:
        raw=(ROOT.parent/rel).read_bytes();published=raw.decode()
        for before,after in SUBSTITUTIONS:published=published.replace(before,after)
        dest=target.parent/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(published)
        notices[rel]=dict(source_sha256=sha(raw),published_sha256=sha(published.encode()))
    (target/'ACTIVE_NOTICES_EXPORT.json').write_text(json.dumps(dict(files=notices,
        note='Only active status and acceptance notices are updated outside performance; original inference source/results remain preserved.'),indent=2)+'\n')
    # The repository manifest is a flat map; preserve its existing convention.
    manifest={}
    for f in sorted(a.repo.rglob('*')):
        if not f.is_file() or '.git' in f.relative_to(a.repo).parts or f==a.repo/'PUBLIC_MANIFEST.json':continue
        if '__pycache__' in f.parts or f.suffix=='.pyc':continue
        manifest[str(f.relative_to(a.repo))]=sha(f.read_bytes())
    (a.repo/'PUBLIC_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(entries),destination=str(target),repository_manifest_files=len(manifest))))

if __name__=='__main__':main()
