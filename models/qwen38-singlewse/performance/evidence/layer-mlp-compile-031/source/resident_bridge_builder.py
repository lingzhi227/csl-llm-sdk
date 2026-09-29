"""Add explicit color translators to the fully admitted P48 original banks."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from spatial.resident_gdn_bridge import compose,weighted_frontend
from spatial.resident_gdn_bridge_routes import lower
from spatial.resident_gdn_routes import emit
from tools.build_frontend_stage import verified_frozen,encoded
from tools.build_resident_gdn_stage import retained_routes
ROOT=Path(__file__).resolve().parents[1]


def build(groups=(15,14)):
    base=ROOT/'evidence/layer-mlp-compile-029';manifest_raw=(base/'source-manifest.json').read_bytes()
    manifest=json.loads(manifest_raw)['files']
    files={n:verified_frozen(base/'source'/n,h)for n,h in manifest.items()if n not in ('execute.py','stager.py')}
    cpath=ROOT/'evidence/resident-gdn-census-002.json';census=json.loads(cpath.read_text())
    if not census['passed']or census['source_manifest_sha256']!=hashlib.sha256(manifest_raw).hexdigest():raise ValueError('Full original compiler census identity')
    original=json.loads(files['profiles.json']);profiles={tuple(pe):dict(source=c['source'],parameters=deepcopy(c['parameters']))for c in original['profiles']for pe in c['pes']}
    placement=json.loads(files['gdn-bank-placement.json']);workers=placement['workers'];existing=json.loads(files['resident-gdn-routes.json'])
    plan=lower(profiles,workers,{tuple(p[:2]):p[2]for p in census['cells']},retained_routes(files),existing['routes'],groups)
    plan['rect']=placement['stage']['rect'];plan['complete_routed_groups']=sorted(existing['groups']+list(groups))
    plan['census_sha256']=hashlib.sha256(cpath.read_bytes()).hexdigest();plan['base_source_manifest_sha256']=hashlib.sha256(manifest_raw).hexdigest()
    for group in plan['plans']:
        for name,pair in [('source_workers',group['incoming']),('child_workers',group['outgoing'])]:
            for pe in group[name]:profiles[tuple(pe)]['parameters'].update(gdn_input_color=pair[0],gdn_output_color=pair[1])
        front=profiles[tuple(group['frontend'])];old=front['source'];changed='weighted_'+old
        if changed not in files:files[changed]=weighted_frontend(files[old])
        front['source']=changed;front['parameters'].update(gdn_send_color=group['incoming'][0],gdn_return_color=group['incoming'][1])
        bridge=profiles[tuple(group['bridge'])];old=bridge['source'];changed='bridge_'+old
        if changed not in files:files[changed]=compose(files[old],old)
        mixer=old=='device_mixer_layer_mlp_standby.csl'
        bridge['source']=changed;bridge['parameters'].update(bridge_group=group['group'],bridge_frames=group['child_frames'],bridge_workers=group['child_markers'],
            bridge_input=group['incoming'][0],bridge_return_output=group['incoming'][1],bridge_output=group['outgoing'][0],bridge_return_input=group['outgoing'][1],
            bridge_forward_queue=2 if mixer else 7,bridge_reverse_queue=3 if mixer else 5)
    classes={};ox,oy,width,height=plan['rect']
    for pe,p in profiles.items():
        key=p['source'],json.dumps(p['parameters'],sort_keys=True)
        classes.setdefault(key,dict(source=p['source'],parameters=p['parameters'],pes=[]))['pes'].append(list(pe))
    old=files['layout.csl'].decode();lines=[old.split(' const program_0=',1)[0].rstrip()]
    for i,c in enumerate(classes.values()):
        name=f'program_{i}';coords=[v for x,y in c['pes']for v in(x-ox,y-oy)]
        fields=','.join('.%s=%s'%(k,v if isinstance(v,str)else str(v).lower())for k,v in c['parameters'].items())
        lines.append(' const %s=[%d]u16{%s};'%(name,len(coords),','.join(map(str,coords))))
        lines.append(' for(@range(u16,%d))|i|{@set_tile_code(%s[2*i],%s[2*i+1],"%s",.{.memcpy_params=ordinary_memcpy_params(%s[2*i]),%s});}'%(len(coords)//2,name,name,c['source'],name,fields))
    suffix=old[old.index(' const route_0='):];at=suffix.index(' @export_name')
    suffix=suffix[:at]+emit(plan)+' @export_name("bridge_arm",fn()void);\n @export_name("bridge_inspect",fn()void);\n @export_name("bridge_status",[*]u32,false);\n'+suffix[at:]
    files['layout.csl']=('\n'.join(lines)+'\n'+suffix).encode()
    original.update(profiles=list(classes.values()),scope=__doc__);files['profiles.json']=encoded(original)
    files['resident-gdn-bridge-routes.json']=encoded(plan)
    files['bridge-ownership.json']=encoded(dict(source_chunks_per_token=9,chunk_words=129,return_frame_words=5,weighted_child_marker=True,
        forward_resources=dict(input_queues={'compact':7,'mixer':2},output_queue=6,dsr=5,ut=7,task=10),
        reverse_resources=dict(input_queues={'compact':5,'mixer':3},output_queue=7,dsr=4,ut=1,task=18,control_task=40,weighted_control_task=41),
        original_banks_unchanged=True,requires_explicit_gdn_phase_arm=True,requires_local_callback_fence_before_native_projection=True,
        serving_controller_integrated=False,neural_execution=False,physical=False))
    files['gdn_bridge.cslpart']=(ROOT/'runtime/gdn_bridge.cslpart').read_bytes()
    files['resident_gdn_bridge.py']=(ROOT/'spatial/resident_gdn_bridge.py').read_bytes()
    files['resident_gdn_bridge_routes.py']=(ROOT/'spatial/resident_gdn_bridge_routes.py').read_bytes()
    files['resident_bridge_builder.py']=Path(__file__).read_bytes()
    files['resident-gdn-bridge-base-census.json']=cpath.read_bytes()
    return files,width,height,[dict(pe=list(pe),**p)for pe,p in profiles.items()]
