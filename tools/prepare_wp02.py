"""Freeze the two-PE reduced operator chain; reuses an existing original tile."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from prepare_wp01 import prepare as prepare_base
from qwen38.chain_numerics import POLICY
from qwen38.resource_ledger import LEDGER,check


def prepare(root,run,image,tile=None):
    check(LEDGER)
    run=prepare_base(root,run,image,tile)
    for name in ('layout.csl','pe.csl','driver.py','compile_entry.py'):
        shutil.copyfile(root/'examples/wp02'/name,run/name)
    (run/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    (run/'resource-ledger.json').write_text(json.dumps(LEDGER,indent=2)+'\n')
    steps=json.loads((run/'steps.json').read_text())
    steps['steps'][0]['argv']=[arg.replace('--fabric-dims=8,3','--fabric-dims=9,3') for arg in steps['steps'][0]['argv']]
    (run/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(run.rglob('*')) if p.is_file() and p.name!='source-manifest.json'}
    (run/'source-manifest.json').write_text(json.dumps({'files':files,'scope':'WP02 two-PE 2x56 GEMV/fabric/SUM/RMS128'},indent=2)+'\n')
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--tile',type=Path)
    args=parser.parse_args();print(prepare(Path(__file__).resolve().parents[1],args.run,args.image,args.tile))


if __name__=='__main__':main()
