"""Collect compact audit receipts and Python sources; leave dense routes remote."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(); p.add_argument('attempt'); p.add_argument('--kind',choices=['projection','columnar','coupled'],default='projection'); a = p.parse_args()
if len(a.attempt) != 3 or not a.attempt.isdigit():
    raise ValueError('Three-digit attempt required')
name = a.kind + '-audit-' + a.attempt
script = 'name=' + repr(name) + '\n' + '''import base64,json
from pathlib import Path
r=Path('/srv/model-storage/qwen38-singlewse/runs')/name
m=json.loads((r/'source-manifest.json').read_text()); sources={}; receipts={}
for n in m['files']:
 p=Path(n);assert not p.is_absolute() and '..' not in p.parts
 if p.suffix=='.py':sources[n]=base64.b64encode((r/n).read_bytes()).decode()
for n in ['source-manifest.json','COMPLETE.json','FAILURE.json','result.json']:
 if (r/n).is_file():receipts[n]=(r/n).read_text()
assert sum(map(len,sources.values()))+sum(map(len,receipts.values()))<524288
print(json.dumps(dict(sources=sources,receipts=receipts)))
'''
r = subprocess.run(['ssh', 'workstation', 'python3 -c ' + shlex.quote(script)], capture_output=True, text=True, check=True, timeout=30)
data = json.loads(r.stdout); out = ROOT / 'evidence' / name
manifest = json.loads(data['receipts']['source-manifest.json'])['files']
files = {n: s.encode() for n, s in data['receipts'].items()}
for n, s in data['sources'].items():
    b = base64.b64decode(s)
    if hashlib.sha256(b).hexdigest() != manifest[n]:
        raise ValueError('Source changed: ' + n)
    files['source/' + n] = b
for n, b in files.items():
    path = out / n; path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != b:
            raise ValueError('Frozen receipt changed: ' + n)
    else:
        path.write_bytes(b)
print(json.dumps(dict(attempt=name, receipts=list(data['receipts']), source_files=len(data['sources']))))
