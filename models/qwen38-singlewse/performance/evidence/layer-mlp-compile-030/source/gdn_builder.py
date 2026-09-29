"""Complete-bank GDN co-placement; recurrent graph routes remain open.

Regenerate original complete MLP routes after row rebalancing. Preserve mixer,
frontend and device-control routes. Real recurrent math/transport is callable
through explicit diagnostic entries, without artificial startup execution.
"""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from spatial.compact_mlp import CompactMlpPlacement
from spatial.gdn_columns import compose_controlled_port
from spatial.gdn_placement import plan,refine_from_census
from spatial.layer_fused_routes import build_fused_routes
from spatial.layer_mlp_network import build_mlp_network,worker_profiles,emit_routes
from tools.build_frontend_stage import verified_frozen,encoded
ROOT=Path(__file__).resolve().parents[1]


def build(refine=False):
 baseline=ROOT/'evidence/layer-mlp-compile-024';manifest=json.loads((baseline/'source-manifest.json').read_text())['files']
 files={n:verified_frozen(baseline/'source'/n,d) for n,d in manifest.items() if n not in ('execute.py','stager.py')}
 previous=json.loads(files['frontend-stage.json']);grouped=json.loads(files['profiles.json'])
 profiles=[dict(pe=pe,source=c['source'],parameters=deepcopy(c['parameters'])) for c in grouped['profiles'] for pe in c['pes']]
 census=json.loads((ROOT/'evidence/compact-mlp-census-001/result.json').read_text())
 placement=plan(previous,json.loads(files['mlp-bank-placement.json']),census,profiles)
 if refine:
  full=ROOT/'evidence/layer-mlp-compile-026'
  full_manifest=json.loads((full/'source-manifest.json').read_text())['files']
  if encoded(placement)!=verified_frozen(full/'source/gdn-bank-placement.json',full_manifest['gdn-bank-placement.json']):raise ValueError('Planning source differs from measured complete program')
  actual=json.loads((ROOT/'evidence/gdn-cohost-census-001.json').read_text())
  if actual['source_manifest_sha256']!=hashlib.sha256((full/'source-manifest.json').read_bytes()).hexdigest():raise ValueError('Complete composed census identity changed')
  placement=refine_from_census(previous,json.loads(files['mlp-bank-placement.json']),placement,actual)
  files['gdn-reservation-census.json']=encoded(actual)
  files['gdn-refinement-source-manifest.json']=(full/'source-manifest.json').read_bytes()
 stage=placement['stage'];compact=CompactMlpPlacement(stage)
 gate=next(r for r in stage['regions'] if r['role']=='gate_up')
 fused=build_fused_routes(gate);network=build_mlp_network(stage,fused,True,True)
 mlp={tuple(p['pe']):p for p in worker_profiles(stage,fused,network)}
 ends={tuple(p['pe']):p['bytes'] for p in placement['bank_ends']}
 workers={tuple(w['pe']):w for w in placement['workers']};proofs={}
 for p in profiles:
  pe=tuple(p['pe'])
  if pe in mlp:p['parameters']=mlp[pe]['parameters']
  if 'bank_words' in p['parameters']:p['parameters']['bank_words']=ends[pe]//4
  if pe in workers:
   w=workers[pe];source=p['source'];changed='gdn_'+source
   if changed not in files:files[changed],proofs[changed]=compose_controlled_port(files[source],source)
   p['source']=changed;p['parameters'].update(gdn_columns=w['columns'],gdn_first=w['first'],gdn_head=w['head'],
    gdn_state_word=w['state_word'],gdn_input_color=0,gdn_output_color=1,gdn_input_queue=7 if source=='compact_layer_projection.csl' else 2)
 # The controller's one retained return frame follows the actual largest sink;
 # all other arenas and the one-response lifetime rule remain unchanged.
 frame_words=network['controller_transport']['frame_words']
 raw=files['mixer_layer_mlp_controller.csl'];anchor=b'var frame=@zeros([273]u32);'
 if raw.count(anchor)!=1:raise ValueError('Controller frame anchor changed')
 files['mixer_layer_mlp_controller.csl']=raw.replace(anchor,('var frame=@zeros([%d]u32);'%frame_words).encode())
 grants=network['grant_schedule'];prep=network['prepare_schedule']
 fields={'targets':'target','kinds':'kind','indices':'index','words':'words','firsts':'first_row','rows':'rows'}
 schedules=['const %s=[%d]u16{%s};'%(n,len(grants),','.join(str(g.get(k,0)) for g in grants)) for n,k in fields.items()]
 schedules+=['const prepare_%s=[%d]u16{%s};'%(n,len(prep),','.join(str(g[k]) for g in prep)) for n,k in {'targets':'target','indices':'index'}.items()]
 files['mlp_schedule.csl']=('\n'.join(schedules)+'\n').encode()
 mp=json.loads(files['mixer-projections.json']);routes=network['routes']+mp['input_routes']
 routes += [dict(pe=r['pe'],color=r['color'],rx=[r['rx']],tx=[r['tx']]) for r in mp['reduction_routes']]
 routes += json.loads(files['device-network.json'])['routes']+json.loads(files['frontend-network.json'])['routes']
 if len({(*r['pe'],r['color']) for r in routes})!=len(routes):raise ValueError('New MLP paths alias retained mixer/control/frontend')
 classes={}
 for p in profiles:
  key=p['source'],json.dumps(p['parameters'],sort_keys=True)
  classes.setdefault(key,dict(source=p['source'],parameters=p['parameters'],pes=[]))['pes'].append(p['pe'])
 old=files['layout.csl'].decode();lines=[old.split(' const program_0=',1)[0].rstrip()]
 for i,c in enumerate(classes.values()):
  name=f'program_{i}';coords=[v for x,y in c['pes'] for v in (x-63,y)]
  fields=','.join('.%s=%s'%(k,v if isinstance(v,str) else str(v).lower()) for k,v in c['parameters'].items())
  lines.append(' const %s=[%d]u16{%s};'%(name,len(coords),','.join(map(str,coords))))
  lines.append(' for(@range(u16,%d))|i|{@set_tile_code(%s[2*i],%s[2*i+1],"%s",.{.memcpy_params=ordinary_memcpy_params(%s[2*i]),%s});}'%(len(coords)//2,name,name,c['source'],name,fields))
 lines.append(emit_routes(dict(rect=stage['rect'],routes=routes)).rstrip())
 lines.append(' @export_name'+old.split(' @export_name',1)[1])
 files['layout.csl']=('\n'.join(lines)+'\n').encode()
 grouped.update(profiles=list(classes.values()),bank_specialization='gdn-bank-placement.json',scope=__doc__)
 files['profiles.json']=encoded(grouped)
 files['previous-worker-setups.json']=files['worker-setups.json']
 files['worker-setups.json']=encoded([[r['pe'],r['setup']] for r in compact.records.values()])
 files['gdn-compact-mlp.json']=encoded(compact.metadata())
 files['gdn-bank-placement.json']=encoded(placement)
 files['gdn-mlp-routes.json']=encoded(fused)
 files['gdn-mlp-network.json']=encoded(network)
 files['gdn-composition.json']=encoded(dict(bindings=proofs,ports_routed=False,physical=False))
 files['previous-network-binding.json']=files['network-binding.json']
 files['network-binding.json']=encoded(dict(stage=stage['id'],audit=network['audit'],senders=network['senders'],sinks=network['down_sinks'],grant_schedule=grants,prepare_schedule=prep,
  distribution_packets=network['distribution_packets'],shared_inputs=True,norm_bridge=True,controller_transport=network['controller_transport'],
  network_sha256=hashlib.sha256(json.dumps(network,sort_keys=True,separators=(',',':')).encode()).hexdigest(),source_bindings={'gdn-mlp-network.json':hashlib.sha256(files['gdn-mlp-network.json']).hexdigest()}))
 files['bank-specialization.json']=encoded(dict(active='gdn-bank-placement.json',logical_metadata='gdn-bank-placement.json:stage',
  auxiliary_namespaces=['gate_up','down','mix'],recurrent_layout='full128-key contiguous value columns',matrix_resolver='CompactMlpPlacement + unchanged CompactMixerPlacement',
  conversations=1,work_slots=2,compiled=False,physical=False,scope=__doc__))
 files['gdn_columns.csl']=(ROOT/'csl/gdn_columns.csl').read_bytes()
 for module in ('gdn_columns','gdn_placement','layer_mlp_network'):files[module+'.py']=(ROOT/'spatial'/(module+'.py')).read_bytes()
 files['gdn_builder.py']=Path(__file__).read_bytes()
 files['p44-source-manifest.json']=(baseline/'source-manifest.json').read_bytes()
 if refine:
  for n in full_manifest:
   if n.endswith('.csl') and n!='layout.csl' and files[n]!=verified_frozen(full/'source'/n,full_manifest[n]):raise ValueError('Auxiliary-only refinement changed actual CSL: '+n)
  before_profiles={tuple(pe):(c['source'],c['parameters']) for c in json.loads(verified_frozen(full/'source/profiles.json',full_manifest['profiles.json']))['profiles'] for pe in c['pes']}
  for p in profiles:
   name,params=before_profiles[tuple(p['pe'])]
   if p['source']!=name or {k:v for k,v in params.items() if k!='bank_words'}!={k:v for k,v in p['parameters'].items() if k!='bank_words'}:raise ValueError('Auxiliary-only refinement changed a program role')
  for n in ('worker-setups.json','mixer-setups.json','gdn-mlp-network.json','network-binding.json'):
   if files[n]!=verified_frozen(full/'source'/n,full_manifest[n]):raise ValueError('Auxiliary-only refinement changed network/descriptors')
 return files,78,146,profiles
