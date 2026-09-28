"""Freeze compact source-bound resource and routing evidence, without allocation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'performance'))
from spatial.frontend_resources import audit_frontend_resources


def build(attempt='023'):
    e=ROOT/'performance/evidence';source=e/'layer-mlp-compile-021/source'
    calibration=e/('layer-backend-compile-'+attempt)
    samples=json.loads((calibration/'source/profiles.json').read_text())['samples']
    profiles=[dict(pe=pe,source=c['source'],parameters=c['parameters'])
              for c in json.loads((source/'profiles.json').read_text())['profiles'] for pe in c['pes']]
    paths=[source/'profiles.json',source/'dialogue-bank-placement.json',
           e/'dialogue-bank-census-001/result.json',calibration/'source/profiles.json',calibration/'sram.json',
           calibration/'source/frontend-network.json',calibration/'source-manifest.json',calibration/'workstation-release.json']
    if not json.loads(paths[-1].read_text())['workstation_released']:raise ValueError('Calibration still owns resources')
    result=audit_frontend_resources(json.loads(paths[5].read_text()),profiles,json.loads(paths[1].read_text()),
                                  json.loads(paths[2].read_text()),samples,json.loads(paths[4].read_text()))
    result['identities']={str(p.relative_to(ROOT/'performance')):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result['calibration']='layer-backend-compile-'+attempt
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
    if not re.fullmatch('[0-9]{3}',a.attempt):raise ValueError('Attempt')
    out=ROOT/'performance/evidence'/('frontend-composition-audit-'+a.attempt)
    if out.exists():raise ValueError('Frozen audit')
    result=build();out.mkdir();(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    files={'audit.py':Path(__file__).read_bytes(),'frontend_resources.py':(ROOT/'performance/spatial/frontend_resources.py').read_bytes()}
    for name,data in files.items():(out/name).write_bytes(data)
    (out/'source-manifest.json').write_text(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['selected_records','estimated_reservations','identities']}))


if __name__=='__main__':main()
