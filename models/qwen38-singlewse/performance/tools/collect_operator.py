"""Collect frozen operator sources and compact receipts, excluding model/ELF/raw arrays."""
import argparse,base64,hashlib,json,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
if not re.fullmatch('(fp8-(native|tile-fast|bank|encoder)|regional-(gemv|bank)|spatial-quant|contraction|mixed-bank|epoch-bank|gdn-bank|route-epoch|filtered-bank|coupled-bank|full-bf16-matrix|native-shapes|retiled-matrix|compact-projection|mlp-fusion|mlp-chunk)-(sim|hw)-[0-9]{3}|retiled-comparison-audit-[0-9]{3}|mlp-(integrated|full)-compile-[0-9]{3}|layer-(backend-compile|weight-audit)-[0-9]{3}',a.name):raise ValueError('name')
physical='-hw-' in a.name
remote=('/srv/qwen38-singlewse-hardware/' if physical else '/srv/model-storage/qwen38-singlewse/runs/')+a.name
script='root_name='+repr(remote)+'\n'+'''import json,base64
from pathlib import Path
r=Path(root_name);m=json.loads((r/'source-manifest.json').read_text())
source={}
for n in m['files']:
 p=Path(n)
 assert not p.is_absolute() and '..' not in p.parts and p.suffix in ['.py','.csl','.json']
 source[n]=base64.b64encode((r/n).read_bytes()).decode()
receipts={n:(r/n).read_text() for n in ['source-manifest.json','COMPLETE.json','FAILURE.json','CANCELLED.json','result.json','sram.json','fixture.json','artifact.json','compile-audit.json','run-audit.json','reuse-artifact.json','reused-compile-audit.json','reuse-compiler-output.json','artifact-admission.json','upload-framing.json','diagnostic.json','static-routes.json','compile.log','run.log','sim.log'] if (r/n).is_file() and (r/n).stat().st_size<1<<20}
# Full-matrix route source plus the exact 1,262-PE ELF census is slightly above
# the original 4MiB envelope. This remains source/compact metadata, never ELF or
# model payload; retain an explicit 8MiB combined transfer limit.
assert sum(len(s) for s in source.values())+sum(len(s) for s in receipts.values())<8388608
print(json.dumps(dict(source=source,receipts=receipts)))
'''
connection=['sh','/path/to/alcf-session.sh','host'] if physical else ['ssh','workstation']
r=subprocess.run(connection+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
data=json.loads(r.stdout);out=ROOT/'evidence'/a.name;out.mkdir(exist_ok=True)
manifest=json.loads(data['receipts']['source-manifest.json'])['files']
for n,s in data['source'].items():
 b=base64.b64decode(s);assert hashlib.sha256(b).hexdigest()==manifest[n]
 f=out/'source'/n;f.parent.mkdir(parents=True,exist_ok=True)
 if f.exists():assert f.read_bytes()==b
 else:f.write_bytes(b)
for n,s in data['receipts'].items():
 f=out/n
 if f.exists():assert f.read_text()==s
 else:f.write_text(s)
print(json.dumps(dict(attempt=a.name,sources=len(manifest),receipts=list(data['receipts']))))
