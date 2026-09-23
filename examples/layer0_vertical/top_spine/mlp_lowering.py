"""Lower the explicitly supported original Layer0 MLP into the existing backend.

The first supported algorithm/shape/precision is deliberately narrow. Spatial
stripe allocation, descriptors and index composition are generated; arithmetic
and the central ready/request protocol remain selected backend primitives.
There is no SDK, weight loading, numerical execution or general graph compiler.
"""
from collections import Counter
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent

def require(ok,field,expected):
    if not ok:raise ValueError(f'Unsupported MLP specification {field}; required {expected}')

def lower(spec):
    require(spec['schema']=='qwen38.current-mlp.v1','schema','qwen38.current-mlp.v1')
    require(spec['layer']==0,'layer','0 in this complete-layer composition')
    require((spec['hidden'],spec['intermediate'])==(5120,17408),'shape','5120 -> 17408 -> 5120')
    require(spec['algorithm']=='down(round_bf16(silu_bf16(gate(x)) * up(x)))','algorithm','the original Gate/Up/SiLU/product/Down composition')
    h,m=spec['hidden'],spec['intermediate'];tile=spec['tiling'];place=spec['placement'];schedule=spec['schedule']
    require(tile==dict(output_rows=128,input_columns=96,product_words=128,zero_pad_input_and_output=True),
            'tiling','128 output rows, 96 input columns and 128-word products, zero padding')
    require(spec['precision']==dict(storage='BF16',input='BF16 expanded exactly to FP32',
        local_accumulator='FP32 FMA in increasing input-column order',
        cross_tile_reduction='FP32 descending logical ordinal chained sum',
        projection_output='BF16 round-to-nearest ties-to-even after full reduction',
        nonlinear='bounded exp backend; whole SiLU rounded BF16 before BF16 product rounding',
        nonlinear_operand_abs_bound=24),'precision','the existing BF16/FP32 ordering and bounded nonlinear backend')
    require(place==dict(application_height=1160,matrix_body_limit=1156,matrix_axis='y',matrix_packing='best-fit-decreasing complete logical chains',
        support_label_base=336,post_norm_label=2,legacy_support_row=1,
        owner_order='ascending product group',projection_order=['gate','up','down']),
        'placement','the explicit vertical-layer placement parameters')
    require(schedule==dict(gate=dict(kind=6,phase=2),up=dict(kind=7,phase=2),down=dict(kind=8,phase=3),
        gate_up_parallel=True,down_ready='all product groups ready',
        product_source_release='copy to retained_chunk then packet source-completion callback',
        matrix_source_release='final row packet source-completion callback',
        math_ready='both Gate and Up complete and packet RX/TX inactive',frame_words=17472),
        'schedule','the current parallel Gate/Up and explicitly released central-frame protocol')
    require(spec['backend']==dict(matrix='matrix_lane.csl',collective='line.csl',nonlinear_owner='mlp_owner.csl',
        nonlinear='nonlinear_kernel.csl',transport='row_packet.csl and existing central frame/packet tree'),
        'backend','the current reusable arithmetic/transport modules')
    rows,cols=tile['output_rows'],tile['input_columns']
    groups=m//rows;owners=[[place['support_label_base']+g,place['legacy_support_row']] for g in range(groups)]
    stripes=[];regions=[]
    for name in place['projection_order']:
        nr,nc=(h,m) if name=='down' else (m,h)
        tensor=spec['tensors'][name]
        require(tensor==dict(id=f'model.language_model.layers.0.mlp.{name}_proj.weight',shape=[nr,nc]),
                f'tensors.{name}','the original tensor identity and matching shape')
        ng=(nr+rows-1)//rows;nt=(nc+cols-1)//cols;kind=schedule[name]['kind'];phase=schedule[name]['phase']
        regions.append(dict(name=tensor['id'],phase=phase,kind=kind,shape=[nr,nc],
                            row_groups=ng,column_tiles=nt,matrix_PEs=ng*nt))
        for group in range(ng):
            valid=min(rows,nr-group*rows)
            target=dict(owner=[place['post_norm_label'],place['legacy_support_row']],offset=group*rows,words=valid) if name=='down' else dict(owner=owners[group],offset=0,words=valid)
            stripes.append(dict(tensor=tensor['id'],kind=kind,phase=phase,group=group,columns=nt,
                                valid_rows=valid,valid_last_columns=nc-(nt-1)*cols,targets=[target]))
    # Intersect the producer chunks and the consumer input tiles in canonical
    # tensor order. One edge is a contiguous shared interval; zero padding has
    # no producer. The current backend realizes these edges via the origin.
    edges=[];blocks=[];cursor=0
    for ordinal in range((m+cols-1)//cols):
        begin=ordinal*cols;end=min(begin+cols,m);first=len(edges)
        while cursor<end:
            group=cursor//rows;stop=min((group+1)*rows,end)
            edges.append(dict(product_group=group,source_offset=cursor-group*rows,
                down_ordinal=ordinal,destination_offset=cursor-begin,words=stop-cursor,
                canonical_begin=cursor,canonical_end=stop))
            cursor=stop
        blocks.append(dict(ordinal=ordinal,canonical_begin=begin,valid_words=end-begin,
                           zero_words=cols-(end-begin),edge_begin=first,edge_end=len(edges)))
    assert cursor==m and len(edges)==272 and len(blocks)==182
    assert Counter(s['kind'] for s in stripes)=={6:136,7:136,8:40}
    return dict(spec_sha256=hashlib.sha256((json.dumps(spec,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest(),
        regions=regions,stripes=stripes,owners=owners,edges=edges,down_input_blocks=blocks,
        down_output_groups=h//rows,placement=place,frame_words=schedule['frame_words'],
        semantics=dict(precision=spec['precision'],schedule=schedule,backend=spec['backend']),
        decisions=dict(automatic=['projection stripe counts and tails','Gate/Up destination owners',
            'Down residual-owner offsets','128-to-96 interval intersections','vertical stripe coordinates through build.py',
            'layout and host descriptors from the resulting stripes'],
            parameterized=['tensor identities/shapes in the supported specification','tile and topology fields validated against the supported backend'],
            manual=['backend kernel selection','descending-ordinal vertical collective','paired endpoint/router placement policy',
                    'central origin and global product-ready barrier','task/queue/DSR assignments in reusable CSL']),
        supported_scope='Original Layer0 5120/17408 BF16 MLP on the vertical backend; other shapes/precision/layout choices diagnose unsupported, not silently ignored.')

def input_offset_expression(lowered,variable):
    """Compile actual edge indices to an affine layout-filter input offset.

    This consumes the derived block/edge representation. The current mapping
    admits ordinal*96; a non-affine mapping cannot silently use that expression.
    """
    assert variable=='ordinal'
    blocks=lowered['down_input_blocks'];edges=lowered['edges']
    stride=blocks[1]['canonical_begin']-blocks[0]['canonical_begin']
    assert blocks[0]['canonical_begin']==0 and stride==96
    for block in blocks:
        assert block['canonical_begin']==block['ordinal']*stride
        position=block['canonical_begin']
        for edge in edges[block['edge_begin']:block['edge_end']]:
            assert edge['canonical_begin']==position
            assert edge['destination_offset']==position-block['canonical_begin']
            assert edge['canonical_end']==edge['canonical_begin']+edge['words']
            position=edge['canonical_end']
        assert position==block['canonical_begin']+block['valid_words']
        assert block['valid_words']+block['zero_words']==stride
    return f'{variable}*{stride}'

def replace_semantics(plan,prior,lowered):
    """Bridge existing non-MLP endpoints without inheriting MLP computation.

    Legacy coordinates are identity keys for the already qualified packet map;
    build.py assigns fresh dense coordinates. All MLP shapes, kinds, phases,
    groups, tails and destinations below come from the specification.
    """
    old_mlp=[s for s in prior['stripes'] if s['kind'] in (6,7,8)]
    assert len(old_mlp)==len(lowered['stripes'])
    generated=[]
    for previous,semantic in zip(old_mlp,lowered['stripes']):
        assert {k:v for k,v in previous.items() if k not in ('x','y')}==semantic
        generated.append(dict(semantic,x=previous['x'],y=previous['y']))
    assert [r for r in prior['matrix_regions'] if r['kind'] in (6,7,8)]==lowered['regions']
    assert prior['support']['MLP_owners']==lowered['owners']
    plan['stripes']=[s for s in plan['stripes'] if s['kind'] not in (6,7,8)]+generated
    plan['matrix_regions']=[r for r in plan['matrix_regions'] if r['kind'] not in (6,7,8)]+lowered['regions']
    plan['support']['MLP_owners']=[list(xy) for xy in lowered['owners']]

def binding_report(lowered,plan):
    stripes=[s for s in plan['stripes'] if s['kind'] in (6,7,8)]
    return dict(scope=lowered['supported_scope'],spec_semantic_sha256=lowered['spec_sha256'],
        source='mlp-spec.json -> mlp_lowering.py -> build.py -> layout.csl / plan.json -> host_mapping.py',
        matrix_regions=lowered['regions'],matrix_binding_fields=['kind','group','x','y','columns','legacy_consumer_label'],
        matrix_bindings=[[s[k] for k in ('kind','group','x','y','columns','legacy_consumer_label')] for s in stripes],
        product_owners=plan['support']['MLP_owners'],
        product_to_down_edges=lowered['edges'],down_input_blocks=lowered['down_input_blocks'],
        edge_delivery='Existing producer -> origin chunk -> global frame -> every Down stripe; no direct-consumer path claimed',
        down_output_groups=lowered['down_output_groups'],layout_input_offset_expression=input_offset_expression(lowered,'ordinal'),
        decisions=lowered['decisions'],semantics=lowered['semantics'],
        evidence=dict(executable_generator=True,generated_CSL_consumes_projection_stripes=True,
            host_descriptors_consume_generated_plan=True,layout_filter_consumes_lowered_edge_indices=True,
            arbitrary_graph_or_shape_support=False,new_CSL_runtime_qualified=False,measured_speedup=False,
            frozen_compile_001_modified=False,full_model=False))
