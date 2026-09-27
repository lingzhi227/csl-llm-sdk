"""Collect source and compact simulator evidence, never ELF/raw simulator state."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
if not re.fullmatch('[0-9]{3}',a.attempt):raise ValueError('attempt')
name='mesh-transport-sim-'+a.attempt;remote='/srv/model-storage/qwen38-singlewse/runs/'+name
script='root_name='+repr(remote)+'\n'+'''import json,base64
from pathlib import Path
r=Path(root_name);manifest=json.loads((r/'source-manifest.json').read_text())
source={n:base64.b64encode((r/n).read_bytes()).decode() for n in manifest['files']}
receipts={n:(r/n).read_text() for n in ['COMPLETE.json','FAILURE.json','result.json','sram.json'] if (r/n).is_file()}
result=dict(source=source,receipts=receipts)
assert sum(len(v) for v in source.values())+sum(len(v) for v in receipts.values())<2097152
print(json.dumps(result))
'''
r=subprocess.run(['ssh','workstation','python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
data=json.loads(r.stdout);out=ROOT/'evidence'/name;manifest=json.loads((out/'source-manifest.json').read_text())['files']
for n,s in data['source'].items():
 b=base64.b64decode(s);assert hashlib.sha256(b).hexdigest()==manifest[n]
 f=out/'source'/n;f.parent.mkdir(parents=True,exist_ok=True)
 if f.exists():assert f.read_bytes()==b
 else:f.write_bytes(b)
for n,s in data['receipts'].items():
 f=out/n
 if f.exists():assert f.read_text()==s
 else:f.write_text(s)
print(json.dumps(dict(attempt=name,sources=len(data['source']),receipts=list(data['receipts']))))
