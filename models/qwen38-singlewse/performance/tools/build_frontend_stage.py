"""Build a complete compact-bank frontend stage from the frozen P40 baseline."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from spatial.compact_mixer import CompactMixerPlacement, compact_csl
from spatial.frontend_placement import plan_frontend_stage, lower_frontend_banks
from spatial.layer_mlp_network import emit_routes
from spatial.mixer_frontend_composition import compose_frontend
from spatial.mixer_projection import lower_mixer_projections, FIELDS

ROOT=Path(__file__).resolve().parents[1]


def encoded(value): return (json.dumps(value,separators=(',',':'))+'\n').encode()


def verified_frozen(path,digest):
    data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
    if actual!=digest:
        # Public snapshots retain execution hashes separately from deliberate
        # site-path adaptation. Bind both identities instead of silently
        # accepting an arbitrary edited baseline.
        export=ROOT/'SOURCE_EXPORT.json'
        entry=json.loads(export.read_text())['files'].get(str(path.relative_to(ROOT)),{}) if export.is_file() else {}
        if entry.get('source_sha256')!=digest or entry.get('published_sha256')!=actual:
            raise ValueError('Frozen source identity changed: '+str(path))
    return data


def build():
    baseline=ROOT/'evidence/layer-mlp-compile-021'
    hashes=json.loads((baseline/'source-manifest.json').read_text())['files']
    files={}
    for name,digest in hashes.items():
        data=verified_frozen(baseline/'source'/name,digest)
        if name not in ('execute.py','stager.py'): files[name]=data
    grouped=json.loads(files['profiles.json'])
    profiles=[dict(pe=pe,source=c['source'],parameters=deepcopy(c['parameters'])) for c in grouped['profiles'] for pe in c['pes']]
    stage=json.loads(files['joint-stage.json']); original=next(r for r in stage['regions'] if r['role']=='mix')
    dialogue=json.loads(files['dialogue-bank-placement.json'])
    calibration={}
    for attempt in ('026','027'):
        root=ROOT/('evidence/layer-backend-compile-'+attempt)
        manifest=json.loads((root/'source-manifest.json').read_text())['files']
        for name,digest in manifest.items():
            verified_frozen(root/'source'/name,digest)
        calibration[attempt]=json.loads((root/'sram.json').read_text())
        files['calibration-'+attempt+'.json']=(root/'sram.json').read_bytes()
        files['calibration-'+attempt+'-manifest.json']=(root/'source-manifest.json').read_bytes()
    samples=json.loads((ROOT/'evidence/layer-backend-compile-026/source/profiles.json').read_text())['samples']
    other=json.loads((ROOT/'evidence/layer-backend-compile-027/source/profiles.json').read_text())['samples']
    if samples!=other: raise ValueError('Compact calibration did not hold all PE parameters fixed')
    files['calibration-samples.json']=encoded(samples)
    files['baseline-census.json']=(ROOT/'evidence/dialogue-bank-census-001/result.json').read_bytes()
    placement=plan_frontend_stage(original,profiles,dialogue,json.loads(files['baseline-census.json']),samples,calibration['026'],calibration['027'])
    banks=lower_frontend_banks(dialogue,placement); region=placement['region']; compact=CompactMixerPlacement(region)
    ends={tuple(p['pe']):p['bytes'] for p in banks['bank_ends']}
    for profile in profiles:
        if 'bank_words' in profile['parameters']: profile['parameters']['bank_words']=ends[tuple(profile['pe'])]//4
    setups=[[r['pe'],r['setup']] for r in compact.records.values() if r['pe']!=stage['request_controller']['pe']]
    # The existing operand and reduction routes depend on complete K groups,
    # which are unchanged. Regenerate their semantic descriptors independently.
    projections=lower_mixer_projections(region,stage['request_controller']['pe'])
    previous=json.loads(files['mixer-projections.json'])
    for field in ('input_routes','reduction_routes'):
        if projections[field]!=previous[field]: raise ValueError('Original contraction routing changed')
    for worker in projections['workers']:
        setup=compact.records[worker['rank']]['setup']
        for m,d in enumerate(worker['descriptors']):
            for i,key in enumerate(FIELDS): d[key]=setup[8*m+i]
            d['scale_base_words']=setup[40+m]
    projections['bank_encoding']='compact-original-mixer-scales-v1'
    projections['audit']['address_audit_scope']='Precompact row/K ownership only; compact addresses verified independently in remote bank proof.'
    compact_csl(files)
    network=compose_frontend(files,profiles,region,setups,native=True,columns=tuple(placement['columns']),batch_rows=True)
    if network!=placement['network']: raise ValueError('Compiled composition differs from planned native routes')
    # Preserve every original router and export declaration verbatim.
    old=files['layout.csl'].decode(); prefix,tail=old.split(' const program_0=',1)
    routes=' const route_0='+tail.split(' const route_0=',1)[1]
    route_part,exports=routes.split(' @export_name',1)
    owned=set()
    for definition,line in zip(route_part.splitlines()[::2],route_part.splitlines()[1::2]):
        values=re.search(r'u16\{([0-9,]+)\}',definition)
        color=re.search(r'@get_color\((\d+)\)',line)
        stride=re.search(r'\[(\d+)\*i\]',line)
        if not (values and color and stride): raise ValueError('Frozen route encoding changed')
        coords=list(map(int,values.group(1).split(',')));step=int(stride.group(1))
        for i in range(0,len(coords),step): owned.add((coords[i]+63,coords[i+1],int(color.group(1))))
    if any((*r['pe'],r['color']) in owned for r in network['routes']):
        raise ValueError('Frontend routes alias existing neural/control traffic')
    classes={}
    for p in profiles:
        key=(p['source'],json.dumps(p['parameters'],sort_keys=True))
        classes.setdefault(key,dict(source=p['source'],parameters=p['parameters'],pes=[]))['pes'].append(p['pe'])
    lines=[prefix.rstrip()]
    for i,c in enumerate(classes.values()):
        name=f'program_{i}';coords=[v for x,y in c['pes'] for v in (x-63,y)]
        fields=','.join('.%s=%s'%(k,v if isinstance(v,str) else str(v).lower()) for k,v in c['parameters'].items())
        lines.append(' const %s=[%d]u16{%s};'%(name,len(coords),','.join(map(str,coords))))
        lines.append(' for(@range(u16,%d))|i|{@set_tile_code(%s[2*i],%s[2*i+1],"%s",.{.memcpy_params=ordinary_memcpy_params(%s[2*i]),%s});}'%(len(coords)//2,name,name,c['source'],name,fields))
    lines.append(route_part.rstrip())
    lines.append(emit_routes(dict(rect=stage['rect'],routes=network['routes'])).replace('route_','frontend_route_').rstrip())
    lines.append(' @export_name'+exports)
    files['layout.csl']=('\n'.join(lines)+'\n').encode()
    stage['regions']=[region if r['role']=='mix' else r for r in stage['regions']]
    for name,value in {'frontend-stage.json':stage,'frontend-placement.json':placement,'frontend-bank-placement.json':banks,
                       'compact-mixer.json':compact.metadata(),'mixer-setups.json':setups,'mixer-projections.json':projections}.items():
        files[name]=encoded(value)
    grouped.update(profiles=list(classes.values()),bank_specialization='frontend-bank-placement.json',
                   scope='Complete original banks with lossless scale aliases, native projection packets and shared convolution/gate frontend. Recurrent core and complete neural execution remain unqualified.')
    files['profiles.json']=encoded(grouped)
    files['bank-specialization.json']=encoded(dict(active='frontend-bank-placement.json',logical_metadata='frontend-stage.json',
        source_dialogue='dialogue-bank-placement.json',source_joint='joint-stage.json',
        auxiliary_resolver='FrontendAuxiliaryPlacement',matrix_resolver='CompactMixerPlacement',
        original_page_ids_preserved=True,conversations=1,work_slots=2,physical=False))
    for name in ('frontend_placement','compact_mixer','mixer_rebalance','mixer_frontend_network','mixer_frontend_composition'):
        files[name+'.py']=(ROOT/'spatial'/(name+'.py')).read_bytes()
    files['frontend_builder.py']=Path(__file__).read_bytes()
    files['baseline-source-manifest.json']=(baseline/'source-manifest.json').read_bytes()
    return files,78,146,profiles
