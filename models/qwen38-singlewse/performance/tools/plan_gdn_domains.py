"""Bounded, local route search from the immutable original-coordinate P49 banks."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_frontend_stage import verified_frozen
from tools.build_resident_gdn_stage import retained_routes
from spatial.gdn_domain_routes import lower


def inputs():
    folder = ROOT/'evidence/layer-mlp-compile-031'
    manifest_raw = (folder/'source-manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)['files']
    names = ['profiles.json', 'gdn-bank-placement.json', 'gdn-mlp-network.json',
             'mixer-projections.json', 'device-network.json', 'frontend-network.json',
             'resident-gdn-routes.json', 'resident-gdn-bridge-routes.json']
    files = {n: verified_frozen(folder/'source'/n, manifest[n]) for n in names}
    profiles = {tuple(pe): c for c in json.loads(files['profiles.json'])['profiles'] for pe in c['pes']}
    workers = json.loads(files['gdn-bank-placement.json'])['workers']
    retained = retained_routes(files)
    for n in ['resident-gdn-routes.json', 'resident-gdn-bridge-routes.json']:
        retained += json.loads(files[n])['routes']
    census_raw = (ROOT/'evidence/gdn-bridge-census-002.json').read_bytes()
    census = json.loads(census_raw)
    assert census['passed'] and census['source_manifest_sha256'] == hashlib.sha256(manifest_raw).hexdigest()
    return profiles, workers, {(x,y):v for x,y,v in census['cells']}, retained


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', default='9,5,6,7,8,10,11,12,13,4,3')
    p.add_argument('--seconds', type=int, default=45)
    p.add_argument('--width', type=int, default=12)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.seconds <= 300 or not 1 <= a.width <= 48:
        p.error('Fresh output, 1..300 seconds and 1..48 candidates required')
    groups = list(map(int, a.groups.split(',')))
    if not set(groups) <= set(range(3,14)):
        p.error('Only unrouted groups 3..13 are allowed')
    progress = []
    def observe(plan):
        progress.append(plan)
        print(json.dumps(dict(group=plan['group'],root_colors=plan['root_colors'],
                              bridges=[c['bridge']for c in plan['children']],
                              routes=len(plan['routes']))),flush=True)
    def expired(*_):
        raise TimeoutError('Bounded offline route search reached its wall limit')
    data = inputs()
    begin = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(a.seconds)
    try:
        result = lower(*data, groups, candidate_width=a.width, on_group=observe)
    except TimeoutError as e:
        result = dict(passed=False,timeout=True,reason=str(e),plans=progress,
                      routes=[r for plan in progress for r in plan['routes']])
    finally:
        signal.alarm(0)
    result.update(groups=groups, candidate_width=a.width,wall_seconds=time.monotonic()-begin,
                  compiled=False,executed=False,physical=False,full_layer=False,
                  remaining_groups=[g for g in range(3,14) if g not in {p['group']for p in result['plans']}],
                  source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in [Path(__file__).resolve(), ROOT/'spatial/gdn_domain_routes.py']})
    with a.output.open('x') as f:
        json.dump(result,f,separators=(',',':'));f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('routes','plans','guard_rejections')}))


if __name__ == '__main__':
    main()
