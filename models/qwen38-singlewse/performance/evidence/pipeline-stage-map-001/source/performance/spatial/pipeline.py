"""Resident layer pipeline: semantic regions -> exact local banks -> 2D stages.

All layers have disjoint physical rectangles. Time reuse is confined to a
region of one layer. This is a placement/kernel IR backend, not an executable
neural implementation or a compiled SRAM/throughput admission.
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, prod

REVISION = '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'


@dataclass(frozen=True)
class PipelineConfig:
    width: int = 750
    height: int = 1160
    macro_columns: int = 8
    concurrency: int = 2
    context: int = 96
    slots: int = 2
    payload_per_pe: int = 35256
    stack_per_pe: int = 4096
    code_sdk_allowance: int = 7424
    communication_scratch: int = 1352
    feedback_rows: int = 2

    def validate(self):
        if (self.width, self.height) != (750, 1160):
            raise ValueError('Pinned physical fabric rectangle required')
        if self.macro_columns not in (4, 8, 16) or self.concurrency < 1:
            raise ValueError('Invalid stage geometry/request capacity')
        if self.context != 96 or self.slots != 2 or self.feedback_rows != 2:
            raise ValueError('First backend uses pinned context96 and two owned slots')
        if min(self.payload_per_pe, self.stack_per_pe, self.code_sdk_allowance,
               self.communication_scratch) <= 0:
            raise ValueError('All resource reservations are required')
        if sum((self.payload_per_pe, self.stack_per_pe, self.code_sdk_allowance,
                self.communication_scratch)) > 48128:
            raise ValueError('PE application SRAM ceiling exceeded')


def bank_profile(fp8, bf16, pages, pes, limit):
    """Exact cyclic tile counts; auxiliary128B pages use remaining local space.

    FP8 tiles are original2x128 bytes plus exact FP32 expansion of their BF16
    scale. BF16 tiles are original1x128 words; paired rows may execute together. Neither is split between PEs.
    Rotating the BF16 stream avoids assuming its remainder is disjoint from the
    FP8 remainder. A bounded interval census counts every PE exactly once.
    """
    f, fr = divmod(fp8, pes); b, br = divmod(bf16, pes)
    start = fr; end = start + br
    cuts = sorted({0, pes, fr, start, min(end, pes), max(0, end-pes)})
    classes = []; capacity = 0; maximum = 0
    for lo, hi in zip(cuts, cuts[1:]):
        if hi == lo:
            continue
        nf = f + (lo < fr)
        nb = b + (((lo-start) % pes) < br)
        base = nf*260 + nb*256
        spare = (limit-base)//128
        classes.append(dict(rank_start=lo, rank_end=hi, fp8_slots=nf,
                            bf16_slots=nb, auxiliary_pages=spare))
        capacity += max(0, spare)*(hi-lo); maximum = max(maximum, base)
    return dict(admitted=maximum <= limit and capacity >= pages,
                matrix_payload_max=maximum, auxiliary_page_capacity=capacity,
                fp8_tiles=fp8, bf16_tiles=bf16, bf16_rank_rotation=start,
                auxiliary_pages=pages, classes=classes)


def minimum_pes(region, config):
    fp, bf, pages = region['fp8_tiles'], region['bf16_tiles'], region['auxiliary_pages']
    lo = max(1, ceil((fp*260+bf*256+pages*128)/config.payload_per_pe))
    hi = config.width*config.height
    while lo < hi:
        mid = (lo+hi)//2
        if bank_profile(fp, bf, pages, mid, config.payload_per_pe)['admitted']:
            hi = mid
        else:
            lo = mid+1
    return lo


def value_bytes(profile):
    return prod(profile['shape'])*{'BF16': 2, 'FP32': 4, 'F32': 4, 'U32': 4}[profile['dtype']]


def semantic_stages(graph, tensors, config):
    if graph['revision'] != REVISION or graph['context'] != config.context:
        raise ValueError('Only the full pinned original graph is supported')
    nodes = graph['nodes']; names = {w for n in nodes for w in n['weights']}
    if len(names) != 1251 or len(nodes) != 1172:
        raise ValueError('Complete original graph required')
    starts = [next(n['id'] for n in nodes if
                   f'model.language_model.layers.{i}.input_layernorm.weight' in n['weights'])
              for i in range(64)] + [1169]
    groups = [('embedding', None, nodes[:1])]
    groups += [(f'layer_{i:02}', i, nodes[starts[i]:starts[i+1]]) for i in range(64)]
    groups += [('head', None, nodes[1169:])]
    stages = []
    for sid, layer, selected in groups:
        if layer is None:
            split = [(sid, selected)]
        else:
            norm = next(n['id'] for n in selected if
                        f'model.language_model.layers.{layer}.post_attention_layernorm.weight' in n['weights'])
            down = next(n['id'] for n in selected if
                        f'model.language_model.layers.{layer}.mlp.down_proj.weight' in n['weights'])
            split = [('mix', [n for n in selected if n['id'] < norm]),
                     ('gate_up', [n for n in selected if norm <= n['id'] < down]),
                     ('down', [n for n in selected if n['id'] >= down])]
        regions = []
        for role, operations in split:
            weights = list(dict.fromkeys(w for n in operations for w in n['weights']))
            scales = {w+'_scale_inv' for w in weights if tensors[w]['dtype'] == 'F8_E4M3'}
            matrices = []; auxiliary = []; fp = bf = macs = 0
            for w in weights:
                t = tensors[w]
                if w in scales:
                    continue
                if len(t['shape']) == 2:
                    m, k = t['shape']
                    if m % 2 or k % 128:
                        raise ValueError('Unlowered matrix shape')
                    kind = t['dtype']; tile_rows = 2 if kind == 'F8_E4M3' else 1
                    count = m//tile_rows*(k//128)
                    if kind not in ('F8_E4M3', 'BF16'):
                        raise ValueError('Unlowered matrix dtype')
                    record = dict(tensor=w, shape=[m, k], tile_shape=[tile_rows, 128], tiles=count,
                                  dtype=kind, stream_start=fp if kind == 'F8_E4M3' else bf)
                    if kind == 'F8_E4M3':
                        scale = tensors[w+'_scale_inv']
                        if scale['shape'] != [ceil(m/128), k//128] or scale['dtype'] != 'BF16':
                            raise ValueError('Original FP8 scale geometry changed')
                        record['scale_tensor'] = w+'_scale_inv'; fp += count
                    else:
                        bf += count
                    # Embedding is a lookup, not a full matrix contraction.
                    macs += m*k if role != 'embedding' else 0
                    matrices.append(record)
                else:
                    auxiliary.append(dict(id=w, kind='weight', bytes=t['bytes']))
            for n in operations:
                a = n['attributes']; op = n['op']
                if op == 'gated_delta_recurrence':
                    per_request = prod(a['state_shape'])*4
                elif op == 'causal_depthwise_conv_silu':
                    per_request = (a['kernel']-1)*a['channels']*2
                elif op == 'kv_append':
                    # Derive the original key and value extents from their inputs.
                    per_request = config.context*sum(value_bytes(graph['values'][v]) for v in n['inputs'])
                else:
                    continue
                auxiliary.append(dict(id=a['state'], kind='request_state',
                                      bytes=per_request*config.concurrency,
                                      bytes_per_request=per_request, requests=config.concurrency))
            # Conservative retention of every distinct input/output of this region
            # in both owned work slots. No liveness savings assumed yet.
            values = sorted({v for n in operations for v in n['inputs']+n['outputs']})
            for v in values:
                auxiliary.append(dict(id=v, kind='slot_value',
                                      bytes=value_bytes(graph['values'][v])*config.slots,
                                      bytes_per_slot=value_bytes(graph['values'][v]), slots=config.slots))
            cursor = 0
            for a in auxiliary:
                a['page_start'] = cursor; a['pages'] = ceil(a['bytes']/128); cursor += a['pages']
            region = dict(id=sid+'/'+role, role=role, nodes=operations, weights=weights,
                          matrices=matrices, fp8_tiles=fp, bf16_tiles=bf, auxiliary=auxiliary,
                          auxiliary_pages=cursor, original_weight_bytes=sum(tensors[w]['bytes'] for w in weights),
                          resident_matrix_bytes=fp*260+bf*256, matrix_macs=macs)
            region['minimum_pes'] = minimum_pes(region, config); regions.append(region)
        stages.append(dict(id=sid, layer=layer, regions=regions,
                           minimum_pes=sum(r['minimum_pes'] for r in regions),
                           matrix_macs=sum(r['matrix_macs'] for r in regions)))
    return stages


@lru_cache(maxsize=4096)
def partition_regions(requirements, macs, width, height):
    """Three adjacent 2D rectangles, with both guillotine orientations explored.

    Full-width operator strips unnecessarily round each role up by a whole row.
    A T partition preserves all three producer/consumer adjacencies and can fit
    the same exact banks with less rectangular waste. No inter-layer overlap.
    """
    if len(requirements) == 1:
        return ((0,0,width,height),) if width*height >= requirements[0] else None
    best = None; best_pressure = float('inf')
    for first in range(3):
        others=[j for j in range(3) if j != first]
        for axis in (0,1):
            size = width if axis == 0 else height
            cross = height if axis == 0 else width
            minimum = ceil(requirements[first]/cross)
            maximum = size-ceil(sum(requirements[j] for j in others)/cross)
            for cut in range(minimum, maximum+1):
                first_rect=(0,0,cut,height) if axis==0 else (0,0,width,cut)
                remaining=(cut,0,width-cut,height) if axis==0 else (0,cut,width,height-cut)
                rx,ry,rw,rh=remaining
                for second_axis in (0,1):
                    length=rw if second_axis==0 else rh; breadth=rh if second_axis==0 else rw
                    low=ceil(requirements[others[0]]/breadth)
                    high=length-ceil(requirements[others[1]]/breadth)
                    if low>high:continue
                    desired=round(length*requirements[others[0]]/sum(requirements[j] for j in others))
                    split=max(low,min(high,desired))
                    rects=[None]*3;rects[first]=first_rect
                    rects[others[0]]=(rx,ry,split,rh) if second_axis==0 else (rx,ry,rw,split)
                    rects[others[1]]=(rx+split,ry,rw-split,rh) if second_axis==0 else (rx,ry+split,rw,rh-split)
                    pressure=max(m/(r[2]*r[3]) for m,r in zip(macs,rects))
                    if pressure<best_pressure:best=tuple(rects);best_pressure=pressure
    return best


def _height(stage, width):
    req=tuple(r['minimum_pes'] for r in stage['regions']);macs=tuple(r['matrix_macs'] for r in stage['regions'])
    h=ceil(sum(req)/width)
    while partition_regions(req,macs,width,h) is None:h+=1
    return h


def place_pipeline(graph, tensors, config=PipelineConfig()):
    config.validate(); stages = semantic_stages(graph, tensors, config)
    available_h = config.height-config.feedback_rows
    count = 64//config.macro_columns
    def group_width(group):
        w = max(1, ceil(sum(s['minimum_pes'] for s in group)/available_h))
        while sum(_height(s, w) for s in group) > available_h:
            w += 1
        return w
    # Unequal numbers of adjacent layers avoid rounding every identical eight-
    # layer macro-column up independently. Dynamic programming preserves order
    # and bounds the stage aspect ratios; it never overlays layers in time.
    choices={(0,0):(0,0,[])}
    for columns in range(config.macro_columns):
        for end in range(65):
            prior=choices.get((columns,end))
            if prior is None:continue
            for length in range(max(1,count-2),count+3):
                if end+length>64:continue
                w=group_width(stages[1+end:1+end+length])
                candidate=(prior[0]+w,prior[1]+(length-count)**2,prior[2]+[(end,end+length,w)])
                key=(columns+1,end+length)
                if key not in choices or candidate[:2]<choices[key][:2]:choices[key]=candidate
    solution=choices[(config.macro_columns,64)][2]
    groups = [[stages[0]]] + [stages[1+start:1+end] for start,end,w in solution] + [[stages[-1]]]
    widths = [group_width(groups[0])] + [w for start,end,w in solution] + [group_width(groups[-1])]
    required_width = sum(widths)
    common = dict(schema='wse-resident-spatial-pipeline-v1', model=graph['model'], revision=graph['revision'],
                  config=vars(config), required_width=required_width, minimum_column_widths=widths.copy(),
                  layers_per_macro_column=[end-start for start,end,w in solution],
                  placement_admitted=required_width <= config.width, compiled_sram_qualified=False,
                  neural_executable=False, physical=False, measured_stage_service_times=None,
                  measured_generated_tokens_per_second=None,
                  resource_allowance_scope='Planning reservations, not composed ELF admission',
                  state_bytes_per_request=sum(a['bytes_per_request'] for s in stages for r in s['regions']
                                              for a in r['auxiliary'] if a['kind'] == 'request_state'))
    if required_width > config.width:
        common.update(rejection='Resident regions plus rectangular rounding exceed fabric width', stages=stages)
        return common
    # Extra columns reduce the largest matrix-MAC/PE pressure proxy. This is not
    # a service-time estimate; replace with measured service distributions later.
    for _ in range(config.width-required_width):
        index = max(range(len(groups)), key=lambda j: max(s['matrix_macs']/max(1,widths[j]*_height(s,widths[j]))
                                                        for s in groups[j]))
        widths[index] += 1
    x = 0
    for column, (group, w) in enumerate(zip(groups, widths)):
        heights = [_height(s, w) for s in group]
        # Distribute spare rows to the largest per-stage compute-pressure proxy.
        for _ in range(available_h-sum(heights)):
            j = max(range(len(group)), key=lambda i: group[i]['matrix_macs']/max(1,w*heights[i]))
            heights[j] += 1
        reverse = 0 < column < len(groups)-1 and (column-1) % 2 == 1
        y = available_h if reverse else 0
        for s, h in zip(group, heights):
            if reverse: y -= h
            s['rect'] = [x, y, w, h]; s['macro_column'] = column
            relative = partition_regions(tuple(r['minimum_pes'] for r in s['regions']),
                                         tuple(r['matrix_macs'] for r in s['regions']),w,h)
            for r, (rx,ry,rw,rh) in zip(s['regions'],relative):
                r['rect'] = [x+rx, y+ry, rw, rh]
                r['banks'] = bank_profile(r['fp8_tiles'],r['bf16_tiles'],r['auxiliary_pages'],rw*rh,config.payload_per_pe)
                if not r['banks']['admitted']:
                    raise ValueError('Actual rectangle failed local bank admission')
            if not reverse: y += h
        x += w
    common.update(stages=stages, column_widths=widths,
                  feedback_corridor=[0, available_h, config.width, config.feedback_rows],
                  stage_order=[s['id'] for s in stages],
                  macro_order='West-to-east columns; adjacent vertical stages snake within each column',
                  required_mean_request_cycle_at_target_seconds=config.concurrency/2000,
                  service_balance_status='Unmeasured; matrix MAC/PE is only an allocation tie-breaker')
    return common


def local_tile_owner(region, matrix, tile):
    """Exact lossless native tile address, independent of any whole-wafer layer ID."""
    if not 0 <= tile < matrix['tiles']:
        raise ValueError('Tile out of range')
    x, y, w, h = region['rect']; n = w*h
    linear = matrix['stream_start']+tile; slot, rank = divmod(linear, n)
    if matrix['dtype'] == 'BF16':
        rank = (rank+region['banks']['bf16_rank_rotation']) % n
    f = region['fp8_tiles']//n + (rank < region['fp8_tiles'] % n)
    offset = slot*260 if matrix['dtype'] == 'F8_E4M3' else f*260+slot*256
    return dict(pe=[x+rank%w, y+rank//w], byte_offset=offset,
                bytes=260 if matrix['dtype'] == 'F8_E4M3' else 256)


def auxiliary_owner(region, page):
    if not 0 <= page < region['auxiliary_pages']:
        raise ValueError('Page out of range')
    x,y,w,h = region['rect']
    for c in region['banks']['classes']:
        capacity = c['auxiliary_pages']*(c['rank_end']-c['rank_start'])
        if page < capacity:
            rank, slot = divmod(page,c['auxiliary_pages']); rank += c['rank_start']
            return dict(pe=[x+rank%w,y+rank//w], byte_offset=c['fp8_slots']*260+c['bf16_slots']*256+slot*128, bytes=128)
        page -= capacity
    raise AssertionError('Admitted page lost')


def audit_pipeline(plan, graph, tensors):
    """Independent coverage, pairwise rectangle and per-class capacity checks."""
    expected = Counter(w for w in dict.fromkeys(w for n in graph['nodes'] for w in n['weights']))
    observed = Counter(w for s in plan['stages'] for r in s['regions'] for w in r['weights'])
    if expected != observed:
        raise ValueError('Original tensor omission or duplication')
    ids = [n['id'] for s in plan['stages'] for r in s['regions'] for n in r['nodes']]
    if sorted(ids) != list(range(len(graph['nodes']))):
        raise ValueError('Original operation omission or duplication')
    if not plan['placement_admitted']:
        return dict(original_tensors=1251, original_operations=len(ids), admitted=False,
                    reason=plan['rejection'])
    rects = [s['rect'] for s in plan['stages']]
    for i,(x,y,w,h) in enumerate(rects):
        if not (w>0 and h>0 and x>=0 and y>=0 and x+w<=750 and y+h<=1158):
            raise ValueError('Stage outside data rectangle')
        for a,b,c,d in rects[:i]:
            if x<a+c and a<x+w and y<b+d and b<y+h:
                raise ValueError('Spatial layers overlap')
    if any(b[0]<a[0] for a,b in zip(rects,rects[1:])):
        raise ValueError('Macro layer order moved west')
    max_payload = 0; resident = 0; total_pes = 0
    for s in plan['stages']:
        sx,sy,sw,sh = s['rect']; covered=0;region_rects=[]
        for r in s['regions']:
            x,y,w,h = r['rect']; total_pes += w*h
            if not (sx<=x and sy<=y and x+w<=sx+sw and y+h<=sy+sh):raise ValueError('Region outside stage')
            for a,b,c,d in region_rects:
                if x<a+c and a<x+w and y<b+d and b<y+h:raise ValueError('Region overlap')
            region_rects.append(r['rect']);covered+=w*h
            classes=r['banks']['classes']; rank=0; fp=bf=pages=0
            for c in classes:
                if c['rank_start'] != rank:raise ValueError('Bank census gap')
                count=c['rank_end']-rank; rank=c['rank_end']
                fp+=count*c['fp8_slots'];bf+=count*c['bf16_slots'];pages+=count*c['auxiliary_pages']
                payload=c['fp8_slots']*260+c['bf16_slots']*256+c['auxiliary_pages']*128
                if payload>plan['config']['payload_per_pe'] or min(c['fp8_slots'],c['bf16_slots'],c['auxiliary_pages'])<0:
                    raise ValueError('Bank overflow')
                max_payload=max(max_payload,payload)
            if (rank,fp,bf)!=(w*h,r['fp8_tiles'],r['bf16_tiles']) or pages<r['auxiliary_pages']:
                raise ValueError('Bank cardinality mismatch')
            resident+=fp*260+bf*256+r['auxiliary_pages']*128
        if covered!=sw*sh:raise ValueError('Stage regions do not cover stage')
    return dict(admitted=True,original_tensors=1251,original_operations=len(ids),stages=66,layers=64,
                original_weight_bytes=sum(tensors[n]['bytes'] for n in expected),
                resident_payload_bytes=resident,data_pes=total_pes,maximum_reserved_payload=max_payload,
                state_bytes_per_request=plan['state_bytes_per_request'], compiled_sram_qualified=False)
