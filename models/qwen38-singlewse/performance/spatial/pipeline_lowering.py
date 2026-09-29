"""Generate layer-local kernel bindings and adjacent spatial stream contracts.

Resolved one-hop boundary paths are separated from required region-internal
distribution. No fabric color/queue admission or executable neural body is
implied by this IR. Numeric modules are reused, not silently requalified.
"""
from .model_subgraph import mlp_subgraph


def boundary_ports(source, destination, max_lanes=40):
    x,y,w,h=source;a,b,c,d=destination
    if x+w==a or a+c==x:
        lo=max(y,b);hi=min(y+h,b+d)
        if hi<=lo:raise ValueError('No shared boundary')
        direction='E' if x+w==a else 'W'
        coords=[(lo+(2*i+1)*(hi-lo)//(2*min(max_lanes,hi-lo))) for i in range(min(max_lanes,hi-lo))]
        return [dict(source=[x+w-1 if direction=='E' else x,v],
                     destination=[a if direction=='E' else a+c-1,v],direction=direction) for v in coords]
    if y+h==b or b+d==y:
        lo=max(x,a);hi=min(x+w,a+c)
        if hi<=lo:raise ValueError('No shared boundary')
        direction='S' if y+h==b else 'N'
        coords=[(lo+(2*i+1)*(hi-lo)//(2*min(max_lanes,hi-lo))) for i in range(min(max_lanes,hi-lo))]
        return [dict(source=[v,y+h-1 if direction=='S' else y],
                     destination=[v,b if direction=='S' else b+d-1],direction=direction) for v in coords]
    raise ValueError('Nonadjacent rectangles need explicit routing')


def lower_pipeline(plan, graph, tensors):
    if not plan['placement_admitted']:raise ValueError('Reject unplaced kernel lowering')
    stages=plan['stages']; producer={v:n for n in graph['nodes'] for v in n['outputs']}
    owner={n['id']:r for s in stages for r in s['regions'] for n in r['nodes']}
    streams=[]; kernels=[]
    for index,stage in enumerate(stages):
        local=[]
        for r in stage['regions']:
            local.append(dict(region=r['id'],rect=r['rect'],original_nodes=[n['id'] for n in r['nodes']],
                              original_operations=[n['op'] for n in r['nodes']],
                              matrices=r['matrices'],banks=r['banks'],auxiliary=r['auxiliary'],
                              tile_ownership='spatial.pipeline.local_tile_owner',
                              state_ownership='spatial.pipeline.auxiliary_owner',
                              request_state_byte_offset='request_id * bytes_per_request + element_byte_offset',
                              allowed_temporal_reuse='Only resident matrices of this one region and layer',
                              csl_neural_body_status='Requires composition and compiled SRAM qualification'))
        fusions=[]
        if stage['layer'] is not None:
            spec=mlp_subgraph(graph,tensors,stage['layer'])
            norm,gate,up,activation,down,residual=spec['original_nodes']
            fusions=[dict(name='residual_rms_quant_fork',region=stage['id']+'/gate_up',
                          source_nodes=[norm['id'],gate['id'],up['id']],
                          consumer='Both original gate/up native tile schedules',
                          readiness='All5120 BF16 residual values before RMS inverse; each normalized128 group before max/encode',
                          retained='Residual remains leased through down+add; normalized packet shared by both projections',
                          reuse=['../csl/qwen_math.csl','csl/mlp_fused.csl']),
                     dict(name='gate_up_silu_mul_quant_down',region=stage['id']+'/gate_up',
                          source_nodes=[gate['id'],up['id'],activation['id'],down['id']],
                          consumer=stage['id']+'/down',
                          readiness='Both complete-K BF16 rows -> SiLU BF16 -> multiply BF16; all128 multiplied values -> max/encode',
                          retained='Two credited group slots; fanout operand released after both send completions',
                          reuse=['csl/mlp_fused.csl','runtime/mlp_activation.csl']),
                     dict(name='down_residual_successor',region=stage['id']+'/down',
                          source_nodes=[down['id'],residual['id']],
                          consumer=stages[index+1]['id'],
                          readiness='All136 down K blocks accumulated before BF16; retained residual add to BF16; stream128 groups',
                          retained='Outgoing slot until local send callback AND destination consumption credit',
                          reuse=['runtime/mlp_join.csl','runtime/mlp_norm.csl'])]
        kernels.append(dict(stage=stage['id'],layer=stage['layer'],rect=stage['rect'],regions=local,
                            fusions=fusions,protocol_module='csl/pipeline_lease.csl',
                            protocol_params=dict(requests=plan['config']['concurrency'],context=plan['config']['context'],chunks=1 if index==0 else 40),
                            semantic_precision='Original node attributes are authoritative',
                            neural_executable=False))
        # Direct semantic producer -> consumer edges across the layer's regions.
        edges={}
        for r in stage['regions']:
            for n in r['nodes']:
                for value in n['inputs']:
                    if value not in producer:continue
                    source=owner[producer[value]['id']]
                    if source['id']==r['id'] or not source['id'].startswith(stage['id']+'/'):continue
                    key=(source['id'],r['id'],value)
                    try:
                        ports=boundary_ports(source['rect'],r['rect'])
                    except ValueError:
                        ports=[]  # A legal partition can have nonadjacent semantic peers.
                    edges.setdefault(key,dict(source=source['id'],destination=r['id'],value=value,
                                              shape=graph['values'][value],consumers=[],
                                              boundary_ports=ports,
                                              nonadjacent_route_required=not ports,
                                              region_local_distribution_lowered=False))['consumers'].append(n['id'])
        streams.extend(edges.values())
    boundaries=[]
    for source,destination in zip(stages,stages[1:]):
        boundaries.append(dict(source=source['id'],destination=destination['id'],
                               producer_region=source['regions'][-1]['id'],consumer_region=destination['regions'][0]['id'],
                               boundary_ports=boundary_ports(source['rect'],destination['rect']),
                               vector_dtype='BF16',elements=5120,chunk_elements=128,chunks=40,
                               chunk_to_lane='chunk_index modulo lane_count',
                               packet_words=69,header=['request_id','request_generation','position','lease_id','chunk_index'],
                               payload_words=64,slots=2,
                               region_local_distribution_lowered=False,colors_queues_threads_qualified=False))
    return dict(schema='wse-layer-local-kernel-stream-ir-v1',kernels=kernels,internal_streams=streams,
                stage_boundaries=boundaries,
                feedback=dict(source='head',destination='embedding',payload='Selected original full-vocabulary token ID',
                              corridor=plan['feedback_corridor'],direction='W',
                              required_events=['Head selection complete','Output observation accepted','Feedback slot granted','Token delivered'],
                              next_position='Only after this request receives its actual selected token',
                              physical_host_io_endpoints_verified=False,fabric_route_lowered=False),
                request_isolation=dict(capacity=plan['config']['concurrency'],context=plan['config']['context'],
                                       key=['request_id','request_generation','position'],
                                       mutable_state='Disjoint resident request slices; no slot aliases request state',
                                       reset='Drain sends/receives, obtain all66 stage quiescence/state-clear acknowledgements, then advance generation'),
                completion_contract='Send callback releases transport ownership; consumer credit releases remote buffer ownership; both are required before slot reuse',
                numeric_fusion_status='Declared original BF16 boundaries; new composed reduction orders unqualified',
                neural_executable=False,physical=False)


def audit_boundaries(ir):
    count=0
    for edge in ir['stage_boundaries']+ir['internal_streams']:
        seen=set()
        for p in edge['boundary_ports']:
            a,b=p['source'],p['destination'];key=tuple(a+b)
            if sum(abs(x-y) for x,y in zip(a,b))!=1 or key in seen:
                raise ValueError('Boundary lane is duplicated or not a physical neighbor')
            if not all(0<=v[0]<750 and 0<=v[1]<1158 for v in [a,b]):
                raise ValueError('Boundary lane outside model rectangle')
            seen.add(key);count+=1
    if len(ir['kernels'])!=66 or len(ir['stage_boundaries'])!=65:
        raise ValueError('Full pipeline boundaries required')
    return dict(stages=66,adjacent_stage_interfaces=65,one_hop_port_pairs=count,
                nonadjacent_internal_streams=sum(bool(e.get('nonadjacent_route_required'))for e in ir['internal_streams']),
                complete_fabric_routes_admitted=False,neural_execution=False)
