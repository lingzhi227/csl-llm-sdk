"""Select a real full-dimensional MLP fork/join from the pinned semantic graph.

This is semantic/interface/lifetime lowering only. No physical placement, routes,
CSL composition or timing is admitted by this document.
"""
import copy
from graphlib import TopologicalSorter


def mlp_subgraph(graph, tensors, layer=0, tile_rows=2, tile_columns=128):
    prefix=f'model.language_model.layers.{layer}.'
    def weighted(name):
        matches=[n for n in graph['nodes'] if prefix+name in n['weights']]
        assert len(matches)==1
        return matches[0]
    norm=weighted('post_attention_layernorm.weight');gate=weighted('mlp.gate_proj.weight');up=weighted('mlp.up_proj.weight');down=weighted('mlp.down_proj.weight')
    assert norm['outputs']==gate['inputs']==up['inputs']
    acts=[n for n in graph['nodes'] if n['op']=='silu_multiply' and n['inputs']==gate['outputs']+up['outputs']]
    assert len(acts)==1;activation=acts[0];assert activation['outputs']==down['inputs']
    adds=[n for n in graph['nodes'] if n['op']=='residual_add' and n['inputs']==norm['inputs']+down['outputs']]
    assert len(adds)==1;residual=adds[0]
    chosen=[norm,gate,up,activation,down,residual];ids={n['id'] for n in chosen}
    values=copy.deepcopy({v:graph['values'][v] for n in chosen for v in n['inputs']+n['outputs']})
    assert all(v['dtype']=='BF16' for v in values.values())
    assert values[norm['inputs'][0]]['shape']==[5120]
    assert values[gate['outputs'][0]]['shape']==values[up['outputs'][0]]['shape']==[17408]
    assert values[down['outputs'][0]]['shape']==[5120]
    nodes=copy.deepcopy(chosen);quantizations=[]
    for name,source,projections in [('shared_input',norm['outputs'][0],[gate,up]),('down_input',activation['outputs'][0],[down])]:
        size=values[source]['shape'][0];assert size%128==0
        qvalue='quantized:'+source
        attributes=projections[0]['attributes']
        assert all(p['attributes']==attributes for p in projections)
        quantizations.append(dict(id='quantize:'+name,op='dynamic_group128_quantization',inputs=[source],outputs=[qvalue],weights=[],
                                 attributes=dict(source_dtype='BF16',group=128,convention=attributes['activation'],native_packet='128 FP16 encodings of FP8/256 plus one original FP32 activation scale')))
        values[qvalue]=dict(shape=[size//128,65],dtype='packed-u32-native-packet',bytes=(size//128)*260)
        for n in nodes:
            if n['id'] in {p['id'] for p in projections}:
                n['inputs']=[qvalue];n['source_op']=n['op'];n['op']='fp8_native_projection'
                n['attributes']['input_quantization_producer']='quantize:'+name
    lowered=nodes+quantizations;produced={v:n['id'] for n in lowered for v in n['outputs']}
    external=set(v for n in lowered for v in n['inputs'])-set(produced)
    assert external==set(norm['inputs'])
    dag={str(n['id']):[str(produced[v]) if v in produced else 'interface:input:'+v for v in n['inputs']] for n in lowered}
    dag.update({'interface:input:'+v:[] for v in external})
    dag.update({'interface:output:'+v:[str(produced[v])] for v in residual['outputs']})
    order=list(TopologicalSorter(dag).static_order())
    lifetimes={}
    for value,profile in values.items():
        consumers=[n['id'] for n in lowered if value in n['inputs']]
        if value in residual['outputs']:consumers.append('interface:output:'+value)
        lifetimes[value]=dict(producer=produced.get(value,'external-predecessor'),consumers=consumers,
                              release_after_consumers_complete=consumers,shape=profile['shape'],dtype=profile['dtype'])
    weights={name:copy.deepcopy(tensors[name]) for n in chosen for name in n['weights']}
    projections=[]
    for n in [gate,up,down]:
        name=n['weights'][0];m,k=weights[name]['shape'];assert m%tile_rows==k%tile_columns==0
        projections.append(dict(node=n['id'],tensor=name,shape=[m,k],tile_shape=[tile_rows,tile_columns],native_tiles=(m//tile_rows)*(k//tile_columns),
                                original_quantization=n['attributes'],new_reduction_order_qualified=False))
    tile_counts=[p['native_tiles'] for p in projections]
    consumers=[dict(node=n['id'],op=n['op'],inputs=n['inputs']) for n in graph['nodes'] if n['id'] not in ids and any(v in n['inputs'] for v in residual['outputs'])]
    return dict(schema='wse-real-mlp-subgraph-v1',model=graph['model'],revision=graph['revision'],layer=layer,original_nodes=chosen,lowered_semantic_nodes=lowered,
                input_values=sorted(external),output_values=residual['outputs'],real_successor_interfaces=consumers,values=values,completion_dependencies=dag,topological_order=order,
                buffer_lifetimes=lifetimes,original_weights=weights,projections=projections,
                resource_counts=dict(original_weight_bytes=sum(w['bytes'] for w in weights.values()),native_tiles_total=sum(tile_counts),
                                     gate_up_workers_if_fully_parallel=sum(tile_counts[:2]),down_workers_if_one_tile_per_pe=tile_counts[2],
                                     per_tile_fp32_scale_bytes=sum(tile_counts)*4,
                                     note='Arithmetic population only. Interface/quantization/state/code/stack and global27B bank capacity are not admitted; down may reuse released compute/scratch while original weights stay resident.'),
                critical_path_terms=['external predecessor delivery','global RMS reduction and normalized value production','shared group128 quantization and fork distribution',
                                     'max(gate input/native/reduction/delivery, up input/native/reduction/delivery)','per-output BF16 SiLU then BF16 multiply',
                                     'group128 redistribution and quantization','down input/native/full-K reduction/delivery','retained residual BF16 add','actual successor consumption'],
                required_streams=['normalized value to shared quantizer','shared native packets to both gate and up regions','gate/up BF16 rows directly to paired SiLU/multiply consumers',
                                  'multiply BF16 output groups to down quantizer/owners','quantized groups to down native regions','down BF16 output and retained residual directly to add/next-layer consumer'],
                physical_placement_complete=False,all_routes_lowered=False,compiled_sram_qualified=False,executable=False,physical=False,full_model=False,
                scope='Real complete-dimensional semantic fork/join, quantization sharing, successor interfaces and cross-operator lifetimes. This does not execute the subgraph or count host-prepositioned inputs as upstream production.')
