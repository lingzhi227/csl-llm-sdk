"""Lower production GDN endpoint code against unchanged P45 complete banks.

Without selected groups this reserves endpoint colors only. Optional supported
groups add original-coordinate chains; unsupported groups remain unrouted.
Compiler admission and numerical execution are separate from source lowering.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from spatial.resident_gdn import worker, producer, compact_producer
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


def build(routed_groups=(), refine_banks=False):
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
    if refine_banks:
        from spatial.resident_gdn_placement import refine
        census_path=ROOT/'evidence/resident-gdn-census-001.json'
        census=json.loads(census_path.read_text())
        observed=ROOT/'evidence/layer-mlp-compile-028'
        observed_manifest=json.loads((observed/'source-manifest.json').read_text())['files']
        if census['source_manifest_sha256']!=hashlib.sha256((observed/'source-manifest.json').read_bytes()).hexdigest():
            raise ValueError('Measured resident source identity changed')
        for name in proofs:
            if files[name]!=verified_frozen(observed/'source'/name,observed_manifest[name]):
                raise ValueError('Endpoint differs from its full measured census')
        placement=refine(json.loads(files['frontend-stage.json']),json.loads(files['mlp-bank-placement.json']),
                         placement,census,profiles,routes)
        workers={tuple(w['pe']):w for w in placement['workers']}
        groups=Counter(w['head']//3 for w in workers.values())
        ends={tuple(r['pe']):r['bytes']//4 for r in placement['bank_ends']}
        for p in profiles:
            pe=tuple(p['pe']);params=p['parameters']
            if 'bank_words'in params:params['bank_words']=ends[pe]
            if pe in workers:
                w=workers[pe]
                if not p['source'].startswith('resident_gdn_'):
                    p['source']='resident_gdn_'+p['source']
                    if p['source']not in files:raise ValueError('Unmeasured cohost source family')
                incoming,returning=ENDPOINT_COLORS[w['role']]
                if any((*pe,color)in occupied for color in (incoming,returning)):
                    raise ValueError('New state endpoint displaces a retained route')
                params.update(gdn_columns=w['columns'],gdn_first=w['first'],gdn_head=w['head'],gdn_state_word=w['state_word'],
                              gdn_input_color=incoming,gdn_output_color=returning,gdn_input_queue=2 if w['role']=='mix'else 7)
            elif 'frontend_group'in params:
                params['gdn_worker_count']=groups[params['frontend_group']]
        name='resident_frontend_native_001_device_mixer_layer_mlp_standby.csl'
        files[name]=compact_producer(files[name])
        cost=ROOT/'evidence/layer-backend-compile-037'
        cm=json.loads((cost/'source-manifest.json').read_text())['files']
        if files[name]!=verified_frozen(cost/'source'/('compact_'+name),cm['compact_'+name]):
            raise ValueError('Frontend differs from measured compact producer')
        proofs[name]['adapted_cohost_sha256']=hashlib.sha256(files[name]).hexdigest()
        proofs[name]['outstanding_debt_counters']=True
        files['gdn-bank-placement.json']=encoded(placement)
        files['resident-gdn-source-census.json']=census_path.read_bytes()
        files['resident_gdn_placement.py']=(ROOT/'spatial/resident_gdn_placement.py').read_bytes()
    route_plan = None
    if routed_groups:
        from spatial.resident_gdn_routes import lower, emit, COLORS
        frontends = {p['parameters']['frontend_group']:p['pe'] for p in profiles
                     if 'frontend_group' in p['parameters']}
        route_plan = lower(placement['stage'], list(workers.values()), frontends, routes, routed_groups)
        for p in profiles:
            params = p['parameters']
            if 'gdn_head' in params and params['gdn_head']//3 in routed_groups:
                incoming, returning = COLORS[params['gdn_head']//3]
                params.update(gdn_input_color=incoming, gdn_output_color=returning)
            elif params.get('frontend_group') in routed_groups:
                incoming, returning = COLORS[params['frontend_group']]
                params.update(gdn_send_color=incoming, gdn_return_color=returning)
        files['resident-gdn-routes.json'] = encoded(route_plan)
        files['resident_gdn_routes.py'] = (ROOT/'spatial/resident_gdn_routes.py').read_bytes()
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
    suffix = old[route_start:]
    if route_plan:
        export_start = suffix.index(' @export_name')
        suffix = suffix[:export_start]+emit(route_plan)+suffix[export_start:]
    files['layout.csl'] = ('\n'.join(lines)+'\n'+suffix).encode()
    grouped.update(profiles=list(classes.values()), scope=__doc__)
    files['profiles.json'] = encoded(grouped)
    files['resident-gdn-composition.json'] = encoded(dict(
        baseline='layer-mlp-compile-027',
        base_source_manifest_sha256=hashlib.sha256((baseline/'source-manifest.json').read_bytes()).hexdigest(),
        bindings=proofs, endpoints=len(workers), frontend_groups=len(groups),
        banks_unchanged=not refine_banks, matrix_prefixes_unchanged=True,
        original_routes_unchanged=True, new_ports_routed=bool(route_plan),
        state_auxiliary_refined=refine_banks,
        routed_groups=list(routed_groups), all_groups_routed=False,
        original_bank_bytes=placement['allocated_bytes'], application=[width,height],
        arithmetic_unchanged=True, executed=False, physical=False, full_layer=False))
    files['resident_gdn.py'] = (ROOT/'spatial/resident_gdn.py').read_bytes()
    files['gdn_fusion.py'] = (ROOT/'spatial/gdn_fusion.py').read_bytes()
    files['resident_gdn_builder.py'] = Path(__file__).read_bytes()
    return files, width, height, profiles
