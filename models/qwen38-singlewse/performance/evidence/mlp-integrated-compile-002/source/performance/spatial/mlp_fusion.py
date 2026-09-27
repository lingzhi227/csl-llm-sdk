"""Restricted fusion decisions for the real Qwen MLP, not generic op deletion.

Fusion is represented as arithmetic co-location, direct packet consumption and
bounded cooperative regions. BF16 rounding and group128 maxima remain observable
semantic boundaries even when their intermediate tensors have no global storage.
"""


def fusion_contract(semantic):
    n={v['op']:v for v in semantic['original_nodes'] if v['op']!='fp8_projection'}
    norm=n['zero_centered_rmsnorm'];activation=n['silu_multiply'];residual=n['residual_add']
    gate,up,down=semantic['projections']
    original_outputs={v['id']:v['outputs'][0] for v in semantic['original_nodes']}
    return dict(schema='wse-mlp-fusion-v1',
                decisions=[
                    dict(id='rms-quant-fork',kind='same-owner arithmetic plus multicast',
                         nodes=[norm['id'],'quantize:shared_input'],owners=40,items_per_owner=128,
                         consumes='actual predecessor BF16 output; retain original for residual',
                         produces='one65-word original group128 packet, forked to both projections',
                         preserves=['FP32 global RMS with epsilon1e-6','zero-centered gain','normalized BF16 rounding',
                                    'group128 max clamp1e-10, divide by448, nearest-even E4M3FN'],
                         no_global_materialization=norm['outputs'],
                         trigger='global RMS ready for this token',
                         barrier='original5120-value RMS; per-group maximum; not an extra model-wide barrier'),
                    dict(id='paired-projection-activation',kind='packed direct edge plus same-actor epilogue',
                         nodes=[gate['node'],up['node'],activation['id']],owners=8704,items_per_owner=2,
                         source_epilogue='each projection root rounds two FP32 sums to BF16 and packs one u32',
                         edge_payload='one u32 gate plus one u32 up per pair, adjacent opposing neighbors',
                         consumer_epilogue='expand gate; SiLU; BF16 round; multiply BF16 up; BF16 round; local absmax',
                         preserves=['gate BF16','up BF16','SiLU BF16 before multiply','multiply BF16'],
                         no_global_materialization=[original_outputs[gate['node']],original_outputs[up['node']]],
                         trigger='both actual root packets arrived for this pair',
                         barrier='two local operands only; no17408-output vector barrier'),
                    dict(id='activation-quant-down',kind='cooperative128-value fusion and direct native packet',
                         nodes=[activation['id'],'quantize:down_input',down['node']],owners=136,items_per_owner=128,
                         consumer_epilogue='64 pairs contribute local max; scale broadcast; each pair encodes one packed FP16/256 word',
                         edge_payload='64 encoded u32 words plus original FP32 scale; consumed directly by native down kernel',
                         no_global_materialization=activation['outputs'],
                         trigger='this128-value group and its route-readiness acknowledgements complete',
                         barrier='one original128-value quantization group, not all136 groups',
                         release='consumer native completion and outgoing send callback before packet/scratch reuse'),
                    dict(id='down-residual-successor',kind='credited chunk stream plus same-owner epilogue',
                         nodes=[down['node'],residual['id']],owners=40,items_per_owner=128,
                         consumer_epilogue='round final FP32 down sum to BF16; add retained BF16 residual; BF16 round',
                         edge_payload='bounded FP32 chunks until complete K136 reduction, then BF16 successor chunks',
                         no_global_materialization=[original_outputs[down['node']]],
                         trigger='all original K contributions for this output chunk',
                         barrier='full K136 per chunk; next RMS needs global5120 sum, but local squares may start on arrival',
                         actual_successors=semantic['real_successor_interfaces'])],
                never_eliminate=['original weight/activation quantization scales','specified BF16 rounding',
                                 'all original K contributions','send-completion and successor-consumption ownership'],
                fp8_scale_implementation='Preserve existing CSL FP32(maximum * FP32(1/448)); literal FP32(maximum /448) is a separately reported oracle and can differ by1ULP. This fusion does not silently change that convention or certify original full-model numerical alignment.',
                metrics_required=['complete subgraph latency','queue backpressure and callback stalls',
                                  'tensor payload and control traffic per directed link','live bytes and code/stack per role',
                                  'upstream-to-successor timing including redistributions'],
                implemented_csl_helpers='csl/mlp_fused.csl',
                complete_fused_runtime=False, numeric_qualification=False, measured_speedup=None)
