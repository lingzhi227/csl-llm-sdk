"""Admit production GDN endpoint code against unchanged P45 complete banks.

No new fabric routes are installed here. Regional free colors are reserved for
the endpoint resource census, not evidence of a connected production graph.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from spatial.resident_gdn import worker, producer
from tools.build_frontend_stage import verified_frozen, encoded

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_COLORS = {'gate_up': (18, 20), 'down': (11, 12), 'mix': (5, 4)}


def retained_routes(files):
    routes = deepcopy(json.loads(files['gdn-mlp-network.json'])['routes'])
    mixer = json.loads(files['mixer-projections.json'])
    routes += deepcopy(mixer['input_routes'])
    routes += [dict(pe=r['pe'], color=r['color'], rx=[r['rx']], tx=[r['tx']])
               for r in mixer['reduction_routes']]
    for name in ('device-network.json', 'frontend-network.json'):
        routes += json.loads(files[name])['routes']
    if len({(*r['pe'], r['color']) for r in routes}) != len(routes):
        raise ValueError('Retained route collision')
    return routes


def build():
    baseline = ROOT/'evidence/layer-mlp-compile-027'
    manifest = json.loads((baseline/'source-manifest.json').read_text())['files']
    files = {n: verified_frozen(baseline/'source'/n, digest)
             for n, digest in manifest.items() if n not in ('execute.py', 'stager.py')}
    routes = retained_routes(files)
    occupied = {(*r['pe'], r['color']) for r in routes}
    placement = json.loads(files['gdn-bank-placement.json'])
    workers = {tuple(w['pe']): w for w in placement['workers']}
    groups = Counter(w['head']//3 for w in workers.values())
    grouped = json.loads(files['profiles.json'])
    profiles = [dict(pe=pe, source=c['source'], parameters=deepcopy(c['parameters']))
                for c in grouped['profiles'] for pe in c['pes']]
    proofs = {}
    for p in profiles:
        pe = tuple(p['pe'])
        if pe in workers:
            w = workers[pe]; old = p['source']; base = old.removeprefix('gdn_')
            changed = 'resident_'+old
            if changed not in files:
                files[changed], proofs[changed] = worker(files[base], files[old], base)
            p['source'] = changed
            incoming, returning = ENDPOINT_COLORS[w['role']]
            p['parameters'].update(gdn_input_color=incoming, gdn_output_color=returning)
            if any((*pe, color) in occupied for color in (incoming, returning)):
                raise ValueError('GDN endpoint color occupied')
        elif 'frontend_group' in p['parameters']:
            old = p['source']; changed = 'resident_'+old
            if changed not in files:
                files[changed], files['resident_frontend_native.csl'], proofs[changed] = producer(
                    files[old], files['mixer_frontend_native.csl'])
                files[changed] = files[changed].replace(b'"mixer_frontend_native.csl"', b'"resident_frontend_native.csl"')
                proofs[changed]['adapted_cohost_sha256'] = hashlib.sha256(files[changed]).hexdigest()
            p['source'] = changed
            p['parameters'].update(gdn_worker_count=groups[p['parameters']['frontend_group']],
                                   gdn_send_color=5, gdn_return_color=4)
            if any((*pe, color) in occupied for color in (5, 4)):
                raise ValueError('Frontend endpoint color occupied')
    classes = {}
    for p in profiles:
        key = p['source'], json.dumps(p['parameters'], sort_keys=True)
        classes.setdefault(key, dict(source=p['source'], parameters=p['parameters'], pes=[]))['pes'].append(p['pe'])
    # Preserve the complete original route/export suffix byte-for-byte.
    old = files['layout.csl'].decode()
    route_start = old.index(' const route_0=')
    lines = [old.split(' const program_0=', 1)[0].rstrip()]
    ox, oy, width, height = placement['stage']['rect']
    for i, c in enumerate(classes.values()):
        name = f'program_{i}'; coords = [v for x,y in c['pes'] for v in (x-ox,y-oy)]
        fields = ','.join('.%s=%s' % (k,v if isinstance(v,str) else str(v).lower()) for k,v in c['parameters'].items())
        lines.append(' const %s=[%d]u16{%s};' % (name,len(coords),','.join(map(str,coords))))
        lines.append(' for(@range(u16,%d))|i|{@set_tile_code(%s[2*i],%s[2*i+1],"%s",.{.memcpy_params=ordinary_memcpy_params(%s[2*i]),%s});}' % (len(coords)//2,name,name,c['source'],name,fields))
    files['layout.csl'] = ('\n'.join(lines)+'\n'+old[route_start:]).encode()
    grouped.update(profiles=list(classes.values()), scope=__doc__)
    files['profiles.json'] = encoded(grouped)
    files['resident-gdn-composition.json'] = encoded(dict(
        baseline='layer-mlp-compile-027',
        base_source_manifest_sha256=hashlib.sha256((baseline/'source-manifest.json').read_bytes()).hexdigest(),
        bindings=proofs, endpoints=len(workers), frontend_groups=len(groups),
        banks_unchanged=True, original_routes_unchanged=True, new_ports_routed=False,
        original_bank_bytes=placement['allocated_bytes'], application=[width,height],
        arithmetic_unchanged=True, executed=False, physical=False, full_layer=False))
    files['resident_gdn.py'] = (ROOT/'spatial/resident_gdn.py').read_bytes()
    files['gdn_fusion.py'] = (ROOT/'spatial/gdn_fusion.py').read_bytes()
    files['resident_gdn_builder.py'] = Path(__file__).read_bytes()
    return files, width, height, profiles
