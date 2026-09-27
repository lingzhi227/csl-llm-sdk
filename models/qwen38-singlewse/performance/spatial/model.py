"""Import the original semantic graph with explicit same-token and state edges.

This is semantic/region IR, not an executable full-model backend. It retains all
arithmetic attributes and weight bindings. Region placement is deliberately
unassigned until there is a checked physical mapping.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

REVISION='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
MODEL='Qwen/Qwen3.8-27B-FP8'


def import_graph(graph, tensors):
    if graph['model']!=MODEL or graph['revision']!=REVISION or tensors['revision']!=REVISION:
        raise ValueError('Original model identity mismatch')
    specs=tensors['tensors'];producer={};last_state={};edges=[];nodes=[];used=set()
    current='embedding';regions={};projection_depth={};op_counts=Counter()
    for node in graph['nodes']:
        nid=node['id']
        if nid!=len(nodes):raise ValueError('Noncanonical node order')
        names=node['weights']
        layers=set()
        for name in names:
            if name not in specs:raise ValueError('Unbound original tensor: '+name)
            match=re.search(r'\.layers\.(\d+)\.',name)
            if match:layers.add(int(match[1]))
        if len(layers)>1:raise ValueError('Cross-layer weight ownership')
        if layers:current='layer.%02d'%next(iter(layers))
        elif node['op']=='zero_centered_rmsnorm' and names==['model.language_model.norm.weight']:current='head'
        deps=set()
        for value in node['inputs']:
            if value=='token':continue
            if value not in producer:raise ValueError('Use before definition: '+value)
            deps.add(producer[value]);edges.append(dict(source=producer[value],target=nid,kind='value',value=value,delay_tokens=0))
        state=node['attributes'].get('state')
        if state:
            if state in last_state:
                deps.add(last_state[state]);edges.append(dict(source=last_state[state],target=nid,kind='state',state=state,delay_tokens=0))
            last_state[state]=nid
        for value in node['outputs']:
            if value in producer or value=='token':raise ValueError('Multiple producers')
            producer[value]=nid
        is_projection=node['op'] in ('fp8_projection','bf16_projection')
        projection_depth[nid]=max((projection_depth[d] for d in deps),default=0)+int(is_projection)
        nodes.append(dict(**node,region=current,dependencies=sorted(deps),placement=None))
        region=regions.setdefault(current,dict(nodes=[],weights=set(),placement=None))
        region['nodes'].append(nid);region['weights'].update(names)
        used.update(names);op_counts[node['op']]+=1
    if (len(used),len(nodes),op_counts['gated_delta_recurrence'],op_counts['full_causal_attention'])!=(1251,1172,48,16):
        raise ValueError('Complete original text graph required')
    if sorted(k for k in regions if k.startswith('layer.'))!=['layer.%02d'%i for i in range(64)]:
        raise ValueError('Missing full layer region')
    # Cross-token recurrence edges are separate from the same-token DAG.
    for state,last in last_state.items():
        first=next(n['id'] for n in nodes if n['attributes'].get('state')==state)
        edges.append(dict(source=last,target=first,kind='state',state=state,delay_tokens=1))
    selected=producer[graph['selected_token']]
    edges.append(dict(source=selected,target=0,kind='token_feedback',value=graph['selected_token'],delay_tokens=1))
    for r in regions.values():
        r['weights']=sorted(r['weights'])
        r['original_weight_bytes']=sum(specs[n]['bytes'] for n in r['weights'])
    weight_bytes=sum(specs[n]['bytes'] for n in used)
    if weight_bytes!=29468003328:raise ValueError('Text weight payload changed')
    count=projection_depth[selected]
    return dict(schema='wse-model-regions-v1',model=MODEL,revision=REVISION,
                context=graph['context'],nodes=nodes,edges=edges,regions=regions,
                metrics=dict(operators=len(nodes),original_tensors=len(used),original_weight_bytes=weight_bytes,
                    op_counts=dict(op_counts),dependent_projection_stages=count,
                    target_tokens_per_second=2000,target_token_budget_us=500,
                    equal_projection_budget_us=500/count,
                    budget_note='Optimistic equal share with all other operators/communication omitted; not a performance estimate'),
                executable_full_model=False,spatial_placement_complete=False)


def load_original(root):
    paths=[Path(root)/'configs'/n for n in ['model-graph.json','tensors.json']]
    result=import_graph(*[json.loads(p.read_text()) for p in paths])
    result['provenance']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return result
