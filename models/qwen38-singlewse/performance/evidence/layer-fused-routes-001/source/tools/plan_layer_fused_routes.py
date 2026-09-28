"""Freeze checked cross-kernel routes and literal CSL lowering, with no device job."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'performance'))
from spatial.layer_fused_routes import build_fused_routes, translate_routes, csl_routes


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    out = args.output
    if out.exists(): raise ValueError('Fresh frozen directory required')
    out.mkdir(parents=True)
    path = ROOT/'performance/evidence/layer-native-schedule-002/layer-schedule.json'
    raw = path.read_bytes(); plan = json.loads(raw); manifest = dict(inputs={str(path.relative_to(ROOT)): hashlib.sha256(raw).hexdigest()}, files={})
    for name in ['spatial/layer_fused_routes.py', 'spatial/layer_routes.py', 'spatial/layer_schedule.py',
                 'runtime/layer_projection.csl', 'csl/layer_fusion_ingress.csl', 'tools/plan_layer_fused_routes.py']:
        contents = (ROOT/'performance'/name).read_bytes(); manifest['files'][name] = hashlib.sha256(contents).hexdigest()
        destination = out/'source'/name; destination.parent.mkdir(parents=True, exist_ok=True); destination.write_bytes(contents)
    (out/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    started = time.monotonic(); templates = {}; audits = []
    try:
        for stage in plan['stages']:
            for region in stage['regions']:
                if region['role'] != 'gate_up': continue
                shape = tuple(region['rect'][2:])
                if shape not in templates:
                    routes = build_fused_routes(region); templates[shape] = (region, routes)
                    name = 'routes-%dx%d' % shape
                    (out/(name+'.json')).write_text(json.dumps(routes, separators=(',', ':'))+'\n')
                    (out/(name+'.csl')).write_text(csl_routes(routes, region['rect'][:2]))
                    print(json.dumps(dict(region=region['id'], seconds=time.monotonic()-started, **routes['audit'])), flush=True)
                else:
                    original, template = templates[shape]; routes = translate_routes(template, original, region)
                audits.append(dict(region=region['id'], rect=region['rect'], **routes['audit']))
        result = dict(passed=True, physical=False, executed=False, seconds=time.monotonic()-started, regions=audits,
                      templates=len(templates), flows=sum(a['flows'] for a in audits),
                      projected_blocks=sum(a['original_projected_blocks'] for a in audits),
                      scope='Static original MLP reduction/projection/credit routes only; input/downstream distributions and full neural composition remain open.')
        (out/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'regions'}), flush=True)
    except BaseException as error:
        (out/'FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__, message=str(error)))+'\n'); raise


if __name__ == '__main__': main()
