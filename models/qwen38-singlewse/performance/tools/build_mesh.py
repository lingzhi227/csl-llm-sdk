"""Deterministic lowering of the checked mesh plan; never compiles or allocates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial.mesh import MeshPlan


def build(out, plan):
    if out.exists():
        raise ValueError('Use a fresh immutable build directory')
    document=plan.document()
    out.mkdir(parents=True)
    files={'layout.csl':plan.emit_layout().encode(),
           'pe.csl':(ROOT/'kernels/mesh_roundtrip.csl').read_bytes()}
    for name,raw in files.items():
        (out/name).write_bytes(raw)
    manifest=dict(plan_sha256=plan.fingerprint(), files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},
                  physical=False, full_model=False, generator_sha256=hashlib.sha256((ROOT/'spatial/mesh.py').read_bytes()).hexdigest())
    (out/'plan.json').write_text(json.dumps(document,indent=2)+'\n')
    (out/'lowering.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,default=16);p.add_argument('--height',type=int,default=16)
    p.add_argument('--max-words',type=int,default=256);a=p.parse_args()
    print(json.dumps(build(a.output,MeshPlan(width=a.width,height=a.height,max_words=a.max_words))))
