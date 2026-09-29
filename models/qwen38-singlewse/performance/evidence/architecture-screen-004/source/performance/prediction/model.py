"""Read emitted placement/stream IR and measured primitive evidence.

Partial native costs are conditional reuse scenarios, not full-model bounds.
Unknown operations never acquire zero-cost status or a fabricated TPS estimate.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from .schedule import simulate


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def native_profiles(evidence, vector_scales=True):
    if not evidence.get('physical') or not evidence.get('all_ordered_outputs_exact'):
        raise ValueError('Qualified physical native evidence required')
    table = {}
    for index, shape in enumerate(evidence['shapes']):
        if shape['vector_scales'] != vector_scales:
            continue
        for kind, dtype in [(0,'F8_E4M3'), (1,'BF16')]:
            records = [c['cycles_per_invocation_by_shape'][index] for c in evidence['cases']
                       if c['kind'] == kind and c['decode_each'] == (1 if kind == 0 else 0)]
            if not records or any(not math.isfinite(v) or v <= 0 for v in records):
                raise ValueError('Invalid/missing calibration samples')
            table[(dtype,shape['rows'],shape['columns'])] = dict(
                min=min(records), median=median(records), max=max(records), samples=len(records))
    return table


def boundary_traffic(ir):
    """Only declared adjacent-stage links. Interior redistribution is unknown."""
    loads = Counter()
    total = 0
    for edge in ir['stage_boundaries']:
        ports = edge['boundary_ports']
        if not ports or edge['chunks'] <= 0 or edge['packet_words'] <= 0:
            raise ValueError('Invalid stream packet/port specification')
        if edge['chunk_to_lane'] != 'chunk_index modulo lane_count':
            raise ValueError('Unsupported lane schedule')
        for chunk in range(edge['chunks']):
            p = ports[chunk % len(ports)]
            src, dst = tuple(p['source']), tuple(p['destination'])
            if sum(abs(a-b) for a,b in zip(src,dst)) != 1:
                raise ValueError('Boundary endpoints must be adjacent')
            loads[src+dst] += edge['packet_words']
            total += edge['packet_words']
    return dict(scope='Adjacent-stage boundary traffic only; excludes interior routing, credits and feedback',
                words_per_complete_pass=total, used_directed_links=len(loads),
                maximum_words_per_boundary_link=max(loads.values(),default=0),
                top_links=[dict(source=list(k[:2]),destination=list(k[2:]),words=v)
                           for k,v in loads.most_common(8)], complete_network=False)


def analyze(plan, ir, calibration, *, target_tps=2000, vector_scales=True):
    if not math.isfinite(target_tps) or target_tps <= 0:
        raise ValueError('Positive finite target required')
    if not plan.get('placement_admitted') or not plan.get('audit',{}).get('admitted'):
        raise ValueError('Placement capacity must be admitted first')
    if plan['revision'] != calibration.get('revision') or plan['model'] != calibration.get('model'):
        raise ValueError('Calibration model/revision mismatch')
    order = [s['id'] for s in plan['stages']]
    if order != [s['stage'] for s in ir['kernels']]:
        raise ValueError('Placement and kernel IR stage order differ')
    for stage,kernel in zip(plan['stages'],ir['kernels']):
        if stage['rect'] != kernel['rect'] or [r['id'] for r in stage['regions']] != [r['region'] for r in kernel['regions']]:
            raise ValueError('Placement and kernel IR regions differ')
        for region, kr in zip(stage['regions'],kernel['regions']):
            if region['rect'] != kr['rect'] or region['matrices'] != kr['matrices']:
                raise ValueError('Placement and kernel IR matrix ownership differ')
    profiles = native_profiles(calibration, vector_scales)
    regions = []
    missing = Counter()
    covered_macs = total_macs = 0
    unknown_ops = Counter()
    calibrated_samples = []
    for stage in plan['stages']:
        for region in stage['regions']:
            pes = region['rect'][2]*region['rect'][3]
            counts = Counter()
            details = []
            # Exact known native workload per PE for the cyclic FP8 tile stream.
            # Matrix-specific starting offsets are retained when adding workloads.
            work = [dict(min=0.0,median=0.0,max=0.0) for _ in range(pes)]
            for node in region['nodes']:
                # Even a projection with measured native tiles still lacks its
                # complete input/reduction/control service time.
                unknown_ops[node['op']] += 1
            for matrix in region['matrices']:
                if region['role'] == 'embedding':
                    missing['embedding lookup/routing'] += 1
                    continue
                m,k = matrix['shape'];macs = m*k;total_macs += macs
                rows,cols = matrix['tile_shape'];key=(matrix['dtype'],rows,cols)
                profile=profiles.get(key)
                # BF16 row-tail placement changed since the measured 2-row probe.
                if profile is None:
                    missing[f'{key[0]} tile {rows}x{cols}'] += 1
                    details.append(dict(tensor=matrix['tensor'],status='uncalibrated',macs=macs))
                    continue
                covered_macs += macs
                base,remainder=divmod(matrix['tiles'],pes)
                start=matrix['stream_start']%pes
                for rank,cost in enumerate(work):
                    invocations=base+(((rank-start)%pes)<remainder)
                    for mode in cost:cost[mode] += invocations*profile[mode]
                counts[str(key)] += matrix['tiles']
                details.append(dict(tensor=matrix['tensor'],status='matching-tile calibration, new composition unmeasured',
                                    native_invocations=matrix['tiles'],macs=macs))
                calibrated_samples.append(key)
            regions.append(dict(id=region['id'],stage=stage['id'],pes=pes,
                                known_native_busy_cycles={mode:max(w[mode] for w in work) for mode in ('min','median','max')},
                                matrices=details,calibrated_tile_counts=dict(counts)))
    ordered=sorted(regions,key=lambda r:r['known_native_busy_cycles']['median'],reverse=True)
    stage_known=[]
    for s in plan['stages']:
        rs=[r for r in regions if r['stage']==s['id']]
        stage_known.append(dict(id=s['id'],
            serialized_region_native_cycles=sum(r['known_native_busy_cycles']['median'] for r in rs),
            maximum_region_native_busy_cycles=max(r['known_native_busy_cycles']['median'] for r in rs)))
    traffic=boundary_traffic(ir)
    return dict(schema='wse-pipeline-performance-screen-v2', model=plan['model'],revision=plan['revision'],
        scope='Evidence-calibrated partial resource-cost screening, NOT a full-model TPS prediction',
        target_single_conversation_response_tokens_per_second=target_tps,
        concurrency=plan['config']['concurrency'],context=plan['config']['context'],
        request_state_bytes=plan['state_bytes_per_request'],
        requirements=dict(maximum_mean_dependent_decode_cycle_seconds=1/target_tps,
                          maximum_summed_response_seconds_per_output_token=1/target_tps,
                          response_metric='Sum of assistant output tokens / sum of complete response durations',
                          response_includes=['Appended prompt processing', 'TTFT', 'Dependent decode', 'Output transport'],
                          response_excludes=['Initial model load', 'Human think time'],
                          explanation='Single continuing conversation. Decode budget is necessary, not sufficient; '
                                      'prefill/TTFT consumes additional response budget. Resident capacity does not relax it.'),
        independent_request_capacity=dict(resident_requests=plan['config']['concurrency'],
                                          predicted_aggregate_tokens_per_second=None,
                                          qualifies_single_conversation_target=False),
        calibration=dict(attempt=calibration['attempt'],clock_hz=calibration.get('clock_hz'),
                         vector_scales=vector_scales,fp8_decode_included=True,
                         sample_range_scope='Observed native case min/max only; not confidence bounds on new kernels',
                         transfer_assumptions='Same native tile implementation, per-PE serialized invocations; region and packet overlap unmeasured'),
        coverage=dict(dense_matrix_macs=total_macs,calibrated_shape_macs=covered_macs,
                      calibrated_shape_mac_fraction=covered_macs/total_macs if total_macs else 0,
                      note='Arithmetic coverage is NOT runtime coverage',
                      uncalibrated_matrix_shapes=dict(missing),
                      operations_without_complete_service_calibration=dict(unknown_ops)),
        regions=regions,stages=stage_known,
        top_native_work_regions=[dict(id=r['id'],pes=r['pes'],cycles=r['known_native_busy_cycles']) for r in ordered[:10]],
        boundary_traffic=traffic,
        unknowns=['New layer-local kernel instruction schedules and composed SRAM',
                  'BF16 row-tail compute and embedding lookup',
                  'GDN/attention state operations, nonlinearities, normalization, quantization and full-head selection',
                  'Interior redistribution, reductions, multicast overlap, link/queue conflicts and completion latency',
                  'Host observation, feedback, request admission, prefill and reset',
                  'Calibrated cycle frequency and end-to-end stage latency/initiation intervals'],
        complete_model_prediction=False,predicted_generated_tokens_per_second=None,
        target_achieved=False,hardware_execution=False)


def service_template(plan, plan_hash):
    return dict(schema='wse-stage-service-scenario-v1',plan_sha256=plan_hash,
                clock_hz=None,clock_source=None,feedback_cycles=None,
                feedback_source=None,
                stages=[dict(id=s['id'],latency_cycles=None,ii_cycles=None,
                             slots=plan['config']['slots'],source=None) for s in plan['stages']],
                note='Fill every field from measurements or labeled assumptions. No unknown cost is zero by default.')


def evaluate_scenario(plan, plan_hash, scenario, *, concurrency=None, warmup=100, samples=1000):
    if scenario.get('plan_sha256') != plan_hash:
        raise ValueError('Service profile does not match this exact placement hash')
    if [s['id'] for s in scenario['stages']] != plan['stage_order']:
        raise ValueError('Complete ordered stage service profile required')
    if any(not s.get('source') for s in scenario['stages']) or not scenario.get('feedback_source'):
        raise ValueError('Every service cost and feedback need an evidence/assumption source')
    if any(s['slots']>plan['config']['slots'] for s in scenario['stages']):
        raise ValueError('Scenario exceeds placed buffer capacity')
    count=1 if concurrency is None else concurrency
    if count>plan['config']['concurrency']:
        raise ValueError('Concurrency exceeds this plan; regenerate and admit a new placement')
    clock=scenario.get('clock_hz')
    if clock is not None and not scenario.get('clock_source'):
        raise ValueError('Clock conversion needs provenance; assumed frequency must be labeled')
    result=simulate(scenario['stages'],count,feedback_cycles=scenario['feedback_cycles'],
                    clock_hz=clock,warmup=warmup,samples=samples)
    result.update(plan_sha256=plan_hash,clock_hz=clock,clock_source=scenario.get('clock_source'),
                  source_scope='User-supplied measured or assumed service parameters; never hardware acceptance',
                  metric_scope=('Single-conversation dependent decode; excludes prefill/TTFT/output transport'
                                if count == 1 else 'Independent-request aggregate throughput; not conversation acceptance'),
                  response_inclusive=False, scenario_concurrency=count,
                  target_achieved=False)
    return result
