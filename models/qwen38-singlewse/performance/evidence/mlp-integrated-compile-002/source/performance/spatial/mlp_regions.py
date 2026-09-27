"""Full-dimensional MLP ownership and WSE-specific protocol lowering.

This candidate has 136 group128 regions, paired gate/up contractions, and forty
independent down-output stripes. It deliberately does not inherit the old atlas.
The compact descriptors are executable by the ownership/traffic audit, not yet
by the device. Missing route, SRAM, numeric and runtime proofs remain explicit.
"""
from dataclasses import asdict, dataclass
from graphlib import TopologicalSorter
from .contraction import tree
from .mlp_fusion import fusion_contract


@dataclass(frozen=True)
class MlpRegionSpec:
    columns: int = 9
    group: int = 128
    native_rows: int = 2
    native_columns: int = 128
    down_chunk_words: int = 8
    maximum_return_stream_words: int = 128


class MlpRegions:
    def __init__(self, semantic, spec=MlpRegionSpec()):
        self.semantic, self.spec = semantic, spec
        if [p['shape'] for p in semantic['projections']] != [[17408, 5120], [17408, 5120], [5120, 17408]]:
            raise ValueError('Pinned full MLP dimensions required')
        if (spec.group, spec.native_rows, spec.native_columns) != (128, 2, 128):
            raise ValueError('This backend preserves the qualified native tile')
        if not 1 <= spec.columns <= 9:
            raise ValueError('Region columns')
        if spec.down_chunk_words not in (2, 4, 8, 16, 32, 64, 128):
            raise ValueError('Exact output group chunking required')
        self.groups, self.k, self.pairs = 136, 40, 64
        self.region_width, self.region_height = 83, 65
        self.header = self.k
        self.rows = (self.groups + spec.columns - 1) // spec.columns
        self.width = spec.columns * self.region_width
        self.height = self.header + self.rows * self.region_height
        if self.width > 750 or self.height > 1160:
            raise ValueError('Full original dimensional geometry does not fit')
        if spec.maximum_return_stream_words < 128:
            raise ValueError('One complete original group128 return exceeds budget')

    def origin(self, group):
        if not 0 <= group < self.groups:
            raise ValueError('Intermediate group')
        return (group % self.spec.columns * self.region_width,
                self.header + group // self.spec.columns * self.region_height)

    def worker(self, group, branch, pair, k):
        if branch not in ('gate', 'up') or not 0 <= pair < 64 or not 0 <= k < 40:
            raise ValueError('Worker coordinate')
        x, y = self.origin(group)
        return x + (40 - k if branch == 'gate' else 42 + k), y + 1 + pair

    def activation(self, group, pair):
        if not 0 <= pair < 64:
            raise ValueError('Output pair')
        x, y = self.origin(group)
        return x + 41, y + pair + 1

    def collector(self, group, output_group):
        if not 0 <= output_group < 40:
            raise ValueError('Down output group')
        x, y = self.origin(group)
        return x + 40 - output_group, y

    def distributor(self, column, k, branch='gate'):
        if not 0 <= column < self.spec.columns or not 0 <= k < 40 or branch not in ('gate', 'up'):
            raise ValueError('Distributor')
        return column * 83 + (40 - k if branch == 'gate' else 42 + k), k

    def original_tiles(self):
        """One record per original matrix tile. No weights or tensor padding."""
        projections = self.semantic['projections']
        for group in range(self.groups):
            for pair in range(64):
                for k in range(40):
                    for branch, projection in zip(('gate', 'up'), projections):
                        yield dict(node=projection['node'], tensor=projection['tensor'],
                                   rows=[128 * group + 2 * pair, 128 * group + 2 * pair + 2],
                                   columns=[128 * k, 128 * k + 128],
                                   xy=self.worker(group, branch, pair, k), phase='gate-up', slot='unassigned',
                                   scale_tensor=projection['tensor']+'_scale_inv', scale_index=[group,k])
                    # Transpose the interpretation of the reused gate coordinates:
                    # forty columns now mean output groups, not contraction K.
                    yield dict(node=projections[2]['node'], tensor=projections[2]['tensor'],
                               rows=[128 * k + 2 * pair, 128 * k + 2 * pair + 2],
                               columns=[128 * group, 128 * group + 128],
                               xy=self.worker(group, 'gate', pair, k), phase='down', slot='unassigned',
                               scale_tensor=projections[2]['tensor']+'_scale_inv', scale_index=[k,group])

    def native_edges(self):
        """Lazily lower every full-width gate/up tree and packed fusion edge."""
        nodes=tree(40)
        for group in range(136):
            for pair in range(64):
                for branch in ('gate','up'):
                    for node in nodes[1:]:
                        yield dict(id=f'native:{group}:{pair}:{branch}:{node["rank"]}',
                                   source=list(self.worker(group,branch,pair,node['rank'])),
                                   destination=list(self.worker(group,branch,pair,node['parent'])),
                                   color=4+2*(node['depth']-1)+node['side'], words=3,
                                   phase='gate-up', kind='native-tree')
                    yield dict(id=f'fused-pair:{group}:{pair}:{branch}',
                               source=list(self.worker(group,branch,pair,0)),
                               destination=list(self.activation(group,pair)),
                               color=16 if branch=='gate' else 17,words=1,
                               phase='gate-up',kind='packed-activation-operand')

    def region(self, group):
        x, y = self.origin(group)
        return dict(id=group, grid=[group % self.spec.columns, group // self.spec.columns],
                    origin=[x, y], extent=[83, 65], intermediate_rows=[128 * group, 128 * group + 128],
                    native_workers=5120, activation_actors=64, down_collectors=40,
                    reused_gate_workers=2560, original_matrix_bytes=3 * 128 * 5120,
                    input_packet_owners='forty shared normalized/quantized groups in header',
                    down_input_owner=list(self.activation(group, 0)))

    def distribution(self):
        """All40 original packets: horizontal header fork, vertical K fanout.

        H and V have different colors and a real relay at each turn; claiming a
        router changes a wavelet's color without such an actor would be invalid.
        """
        streams = []
        for k in range(40):
            turns = [self.distributor(c, k, b) for c in range(self.spec.columns) for b in ('gate', 'up')]
            streams.append(dict(id=f'normalized-packet:{k}', kind='header-fork', color=2,
                                source=[0, k], end=list(max(turns)), consumers=[list(p) for p in turns],
                                words=65, phase='gate-up', producer='shared exact quantization after global RMS'))
            for c in range(self.spec.columns):
                groups = list(range(c, self.groups, self.spec.columns))
                for branch in ('gate', 'up'):
                    source = self.distributor(c, k, branch)
                    end = self.worker(groups[-1], branch, 63, k)
                    streams.append(dict(id=f'input:{c}:{branch}:{k}', kind='vertical-multicast', color=3,
                                        source=list(source), end=list(end), groups=groups,
                                        worker_branch=branch, k=k, words=65, phase='gate-up',
                                        consumers=len(groups)*64, relay_requires_full_packet=True))
        return streams

    def down_edges(self):
        """Typed real down projection partials, not an artificial inter-row flow.

        Each edge carries128 FP32 partials in credited chunks; forty groups stay
        distinct through vertical region contraction and header transposition.
        """
        edges = []
        for group in range(self.groups):
            bx, by = group % self.spec.columns, group // self.spec.columns
            for j in range(40):
                source = self.collector(group, j)
                dest = self.collector(group-self.spec.columns, j) if by else self.distributor(bx, j)
                edges.append(dict(id=f'down-v:{group}:{j}', phase='down', kind='vertical-reduction' if by else 'transpose-to-header',
                                  source=list(source), destination=list(dest), color=13+(by%2), words=128,
                                  output_rows=[128*j, 128*j+128], chunk_words=self.spec.down_chunk_words))
        for bx in range(self.spec.columns):
            for j in range(40):
                source = self.distributor(bx, j)
                dest = self.distributor(bx-1, j) if bx else (0, j)
                edges.append(dict(id=f'down-h:{bx}:{j}', phase='down', kind='horizontal-reduction' if bx else 'residual-consumer',
                                  source=list(source), destination=list(dest), color=17+(bx%2), words=128,
                                  output_rows=[128*j, 128*j+128], chunk_words=self.spec.down_chunk_words))
        return edges

    def reduction_expression(self):
        """Unambiguous FP32 binary ordering, deliberately different from source."""
        columns = []
        for bx in range(self.spec.columns):
            groups = list(range(bx, self.groups, self.spec.columns))
            expression = groups[-1]
            for g in reversed(groups[:-1]):
                expression = [g, expression]
            columns.append(expression)
        expression = columns[-1]
        for c in reversed(columns[:-1]):
            expression = [c, expression]
        return expression

    def phase_program(self):
        """Producer/consumer readiness plus router/queue/buffer transitions.

        These are requirements on the CSL backend, not observed runtime events.
        The row fence covers forwarding-only PEs as well as compute owners.
        """
        events = {
            'external-input': [],
            'norm-local-squares': ['external-input'],
            'norm-global-rms': ['norm-local-squares'],
            'normalized-bf16': ['norm-global-rms'],
            'shared-quantized': ['normalized-bf16'],
            'input-relay-packets': ['shared-quantized'],
            'gate-up-native': ['input-relay-packets'],
            'gate-up-row-delivered': ['gate-up-native'],
            'worker-send-callback-idle': ['gate-up-native'],
            'row-fence-delivered': ['gate-up-row-delivered'],
            'worker-routes-installed': ['row-fence-delivered', 'worker-send-callback-idle'],
            'worker-down-receives-bound': ['worker-routes-installed'],
            'row-readiness-ack': ['worker-down-receives-bound'],
            'silu-bf16-multiply-bf16': ['gate-up-row-delivered'],
            'group-max-and-readiness': ['silu-bf16-multiply-bf16', 'row-readiness-ack'],
            'down-scale-broadcast': ['group-max-and-readiness'],
            'down-encoded-packet': ['down-scale-broadcast'],
            'down-broadcast': ['down-encoded-packet'],
            'down-native': ['down-broadcast'],
            'down-local-gather': ['down-native'],
            'down-vertical-chunks': ['down-local-gather'],
            'down-header-chunks': ['down-vertical-chunks'],
            'down-bf16-residual-bf16': ['down-header-chunks', 'external-input'],
            'successor-consumed': ['down-bf16-residual-bf16'],
            'all-send-callbacks-idle': ['successor-consumed'],
            'next-epoch-initial-bindings': ['all-send-callbacks-idle'],
        }
        list(TopologicalSorter(events).static_order())
        return dict(events=events, runtime_global_barriers=False,
                    granularity='row fence per128-row region; group readiness joins64 row acknowledgements; down chunks depend on local and child data',
                    transitions=[dict(scope='gate worker row', colors=list(range(4,11)),
                                      before='horizontal native tree', after='vertical local output gather',
                                      requires=['row-fence-delivered','worker-send-callback-idle'],
                                      before_ack=['worker-routes-installed','worker-down-receives-bound']),
                                 dict(scope='activation column', colors=[15], before='row acknowledgement', after='scale broadcast',
                                      requires=['row-readiness-ack'], before_maximum_send=True)],
                    initial_bindings=[dict(scope='worker/down gather', queue=7, color=4, before='worker-down-receives-bound'),
                                      dict(scope='activation max/gather', queue=6, color=4, before='each maximum or encoded-payload gather phase')],
                    credits=[dict(scope='local gather', capacity_chunks=1, release='outgoing send callback; reset first queue color on every phase'),
                             dict(scope='down vertical/header', capacity_chunks=1, words=self.spec.down_chunk_words,
                                  release='local+child add, outgoing send callback, then next receive'),
                             dict(scope='shared residual input', capacity_epochs=1, release='actual successor consumption and all outgoing callbacks')],
                    warnings=['A send callback alone is not a region readiness fence.',
                              'An acyclic completion DAG alone does not prove queue binding, forwarding drain or deadlock freedom.',
                              'Epoch reset requires a fabric-wide termination protocol that has not been lowered.'])

    def memory(self):
        # Requirements only; code and stack qualification must use actual ELFs.
        helper_roles=136*(64+40)+2*self.spec.columns*40+40
        return dict(weight_assignment='Original tile coordinate ownership exists; resident slot assignment and complete27B packing do not.',
                    maximum_prior_bank_payload_bytes=35256,
                    layer_matrix_tile_payload_bytes=267386880,
                    layer_expanded_scale_copies_bytes=1044480*4,
                    scale_copy_representation='Original BF16 scale_inv expanded exactly to FP32, one per native tile;64 copies per original matrix scale. This increases resident bytes beyond original tensor bytes.',
                    prior_atlas_comparison=dict(prior_bank_pes=860880,wafer_pes=870000,prior_dedicated_actor_allowance=9120,
                                               candidate_helper_roles=helper_roles,
                                               deficit_if_all_helpers_exclude_weight_banks=helper_roles-9120,
                                               admitted=False,
                                               required='Cohost original banks on sufficient helper roles or recompute complete model packing; do not subtract original weights to fit this layer.'),
                    role_requirements={
                        'gate-bank-worker': dict(bank=35256, packet=260, decoded_or_gather_scratch=512,
                                                 native_output=8, native_children=24, stack_reserve=4096,
                                                 unspecified='code, flags, route words, row acknowledgements, SDK, alignment'),
                        'up-bank-worker': dict(bank=35256, packet=260, decoded_scratch=512, native_output=8,
                                               native_children=24, stack_reserve=4096, unspecified='code/control/SDK/alignment'),
                        'down-collector': dict(local_partial=512, child_chunk=4*self.spec.down_chunk_words,
                                               stack_reserve=4096, unspecified='code/control/SDK/alignment'),
                        'input-residual-owner': dict(retained_original_bf16=256, norm_weights_bf16=256,
                                                    normalized_f32_scratch=512, encoded_packet=260, down_partial=512,
                                                    child_chunk=4*self.spec.down_chunk_words, stack_reserve=4096,
                                                    unspecified='code/control/SDK/global RMS/successor interface/alignment')},
                    aliases=[dict(buffer='worker packet', phases=['gate-up input','down input'],
                                  overwrite_after=['gate-up-native','worker-send-callback-idle','row-fence-delivered']),
                             dict(buffer='decoded scratch', phases=['FP8 decode/native','native reduction send','down decode/native','local output gather'],
                                  overwrite_after='previous phase final consumer AND outgoing callback'),
                             dict(buffer='residual BF16', phases=['RMS reads','residual add reads'],
                                  overwrite_after='successor-consumed and all-send-callbacks-idle')],
                    compiled_sram_qualified=False, full_model_banks_admitted=False)

    def cost(self):
        distribution=self.distribution();edges=self.down_edges()
        distance=lambda a,b:abs(a[0]-b[0])+abs(a[1]-b[1])
        input_hops=max(distance(s['source'],s['end']) for s in distribution)
        # Same-direction links for these stream families are disjoint. The
        # independent auditor expands them; all other families remain excluded.
        traffic={kind:sum(s['words']*distance(s['source'],s.get('destination',s.get('end')))
                          for s in distribution+edges if s['kind']==kind)
                 for kind in sorted({s['kind'] for s in distribution+edges})}
        native_tree=tree(40)
        native_hops=sum(n['rank']-n['parent'] for n in native_tree[1:])
        traffic['native-tree']=136*64*2*3*native_hops
        traffic['packed-activation-operand']=136*64*2
        return dict(known_word_hops_by_family=traffic,
                    gate_up_native_tree_word_hops=136*64*2*3*native_hops,
                    maximum_individual_input_route_hops=input_hops,
                    known_directed_link_words_per_family_max=128,
                    return_parallel_stripes=40, return_stream_words=128,
                    single_collector_alternative_words=5120,
                    down_vertical_software_reduction_depth=self.rows-1,
                    down_horizontal_software_reduction_depth=self.spec.columns-1,
                    chunks_per_output_stripe=128//self.spec.down_chunk_words,
                    measured_cycles=None, predicted_tokens_per_second=None,
                    full_credit_critical_path_modeled=False,
                    missing=['RMS routes','row fences/acknowledgements','group max/scale/encoded gather',
                             'local down broadcast/gather','next-epoch termination','SDK and runtime contention'],
                    scope='Known family word-hop counts only;128 is not an admitted total-link maximum and40x smaller streams is not a40x speedup.')

    def document(self):
        actors=136*(5120+64+40)+2*self.spec.columns*40+40
        return dict(schema='wse-real-mlp-regions-v1', model=self.semantic['model'], revision=self.semantic['revision'],
                    semantic_layer=self.semantic['layer'], spec=asdict(self.spec), application=[self.width,self.height],
                    regions=[self.region(i) for i in range(136)],
                    ownership=dict(gate_up_workers=696320, reused_down_workers=348160, activation_actors=8704,
                                   down_collectors=5440, header_distributors=2*self.spec.columns*40, input_residual_owners=40,
                                   distinct_active_pes=actors, rectangle_pes=self.width*self.height,
                                   matrix_tiles=1044480, original_matrix_bytes=267386880,
                                   original_weights_including_scales_norm=self.semantic['resource_counts']['original_weight_bytes']),
                    logical_input_owners=[dict(group=k, xy=[0,k], original_rows=[128*k,128*k+128],
                                              producer=self.semantic['input_values'][0], retained_for=self.semantic['output_values'][0]) for k in range(40)],
                    projection_tensor_names=[p['tensor'] for p in self.semantic['projections']],
                    native_stream_family=dict(generator='MlpRegions.native_edges',streams=696320,
                                              contractions=17408,k_terms_per_contraction=40,
                                              tree_colors=list(range(4,13)),packed_operand_colors=[16,17],
                                              native_words=3,packed_operand_words=1,
                                              reduction_order='P18 ordered local-left-right preorder K40 binary tree;full MLP numerical order not qualified'),
                    input_distribution=self.distribution(), down_edges=self.down_edges(),
                    down_fp32_reduction_expression=self.reduction_expression(),
                    down_original_sum_order_preserved=False, changed_numeric_order_qualified=False,
                    phase_program=self.phase_program(), fusion=fusion_contract(self.semantic), memory=self.memory(), cost=self.cost(),
                    all_routes_lowered=False, complete_resource_leases=False, compiled_sram_qualified=False,
                    executable=False, physical=False, full_model=False,
                    scope='Complete original5120/17408 ownership, real operator interfaces and selected spatial stream families. This is a candidate lowering, not an executable subgraph, admitted27B placement or performance result.')
