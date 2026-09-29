"""Keep the complete P42 neural fabric and losslessly compact MLP scales."""
from copy import deepcopy
import json
from pathlib import Path
from spatial.compact_mlp import CompactMlpPlacement, compact_mlp_csl, lower_mlp_banks
from tools.build_frontend_stage import verified_frozen, encoded

ROOT=Path(__file__).resolve().parents[1]

def build():
    baseline=ROOT/'evidence/layer-mlp-compile-022'
    manifest=json.loads((baseline/'source-manifest.json').read_text())['files']
    files={n:verified_frozen(baseline/'source'/n,d) for n,d in manifest.items() if n not in ('execute.py','stager.py')}
    stage=json.loads(files['frontend-stage.json']); compact=CompactMlpPlacement(stage)
    previous=json.loads(files['frontend-bank-placement.json']); banks=lower_mlp_banks(previous,compact)
    original_setups={tuple(pe):v for pe,v in json.loads(files['worker-setups.json'])}
    if original_setups!={pe:r['original_setup'] for pe,r in compact.records.items()}:
        raise ValueError('Actual compiled MLP setup differs from compact original descriptors')
    grouped=json.loads(files['profiles.json']); profiles=[]
    ends={tuple(p['pe']):p['bytes'] for p in banks['bank_ends']}
    for c in grouped['profiles']:
        for pe in c['pes']:
            p=dict(pe=pe,source=c['source'],parameters=deepcopy(c['parameters']))
            if tuple(pe) in compact.records:
                if p['source']!='layer_projection.csl': raise ValueError('Unexpected MLP cohost source')
                p['source']='compact_layer_projection.csl';p['parameters']['bank_words']=ends[tuple(pe)]//4
            profiles.append(p)
    compact_mlp_csl(files)
    old=files['layout.csl'].decode();prefix,tail=old.split(' const program_0=',1)
    routes=' const route_0='+tail.split(' const route_0=',1)[1]
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
    lines.append(routes)
    files['layout.csl']=('\n'.join(lines)+'\n').encode()
    grouped.update(profiles=list(classes.values()),bank_specialization='mlp-bank-placement.json',
        scope='Complete P42 fabric/banks with local original MLP scale aliases. Recurrent workers and complete neural execution remain unqualified.')
    files['profiles.json']=encoded(grouped)
    files['original-worker-setups.json']=files['worker-setups.json']
    files['worker-setups.json']=encoded([[r['pe'],r['setup']] for r in compact.records.values()])
    files['compact-mlp.json']=encoded(compact.metadata());files['mlp-bank-placement.json']=encoded(banks)
    files['bank-specialization.json']=encoded(dict(active='mlp-bank-placement.json',logical_metadata='frontend-stage.json',
        source_frontend='frontend-bank-placement.json',source_dialogue='dialogue-bank-placement.json',
        auxiliary_resolver='CompactMlpAuxiliaryPlacement',mixer_resolver='CompactMixerPlacement',mlp_resolver='CompactMlpPlacement',
        original_page_ids_preserved=True,conversations=1,work_slots=2,physical=False))
    files['compact_mlp.py']=(ROOT/'spatial/compact_mlp.py').read_bytes()
    files['compact_mlp_builder.py']=Path(__file__).read_bytes()
    files['p42-source-manifest.json']=(baseline/'source-manifest.json').read_bytes()
    return files,78,146,profiles
