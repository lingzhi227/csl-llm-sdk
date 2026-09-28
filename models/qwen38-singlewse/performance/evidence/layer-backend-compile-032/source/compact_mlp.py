"""Exact local sharing of original MLP128-row scale blocks.

Original FP8 bytes, output groups, K slices and reduction routes do not move.
Each branch retains every local K part's FP32 scale per original128-row block.
Aliased source words still require independent bitwise migration verification.
"""
from bisect import bisect_right
from copy import deepcopy
from spatial.layer_routes import contraction_groups, worker


class CompactMlpPlacement:
    def __init__(self, stage):
        self.records = {}; self.groups = {}; self.regions = {}
        for region in stage['regions']:
            if region['role'] not in ('gate_up', 'down'): continue
            role = region['role']; self.regions[role] = region
            groups = list(contraction_groups(region)); self.groups[role] = groups
            for group in groups:
                for node in group['nodes']:
                    w = worker(region, group, node); p = w['parameters']; a = w['arm']
                    parts, iterations, first = a['parts'], a['iterations'], a['first_output']
                    blocks = (first % 128 + iterations*p['rows']+127)//128
                    data_words = iterations*parts*64
                    branch_words = data_words+blocks*parts
                    if w['original_matrix_bytes'] != iterations*parts*p['branches']*260:
                        raise ValueError('Original MLP prefix is not a complete local branch bank')
                    setup = [parts, iterations, 0, branch_words if p['branches']==2 else 0, first]
                    if tuple(w['pe']) in self.records or max(setup) >= 65536:
                        raise ValueError('Duplicate/oversize compact MLP descriptor')
                    self.records[tuple(w['pe'])] = dict(pe=w['pe'], role=role, rank=w['rank'],
                        rows=p['rows'], columns=p['columns'], branches=p['branches'],
                        parts=parts, iterations=iterations, first_output=first,
                        original_setup=list(a.values()), setup=setup,
                        original_bytes=w['original_matrix_bytes'], bytes=branch_words*p['branches']*4,
                        data_words=data_words, branch_words=branch_words, scale_blocks=blocks)
        if set(self.regions) != {'gate_up', 'down'}: raise ValueError('Complete original MLP required')

    def tile(self, role, matrix, tile):
        region=self.regions[role]; m=region['matrices'][matrix]
        if not 0<=tile<m['tiles']: raise ValueError('Original tile extent')
        row,key=divmod(tile,m['k_blocks']); groups=self.groups[role]
        index=bisect_right([g['output_start'] for g in groups],row)-1
        g=groups[index]; node=g['nodes'][key%g['workers']]
        w=worker(region,g,node); record=self.records[tuple(w['pe'])]
        iteration=row-g['output_start']; part=key//g['workers']
        if not 0<=iteration<record['iterations'] or part>=record['parts']:
            raise ValueError('Original MLP local tile ownership')
        base=matrix*record['branch_words']; block=(row*record['rows'])//128-record['first_output']//128
        return dict(pe=w['pe'], byte_offset=4*(base+(iteration*record['parts']+part)*64), bytes=256,
            scale=dict(pe=w['pe'], byte_offset=4*(base+record['data_words']+block*record['parts']+part), bytes=4))

    def metadata(self):
        records=list(self.records.values())
        return dict(schema='compact-original-mlp-scales-v1', records=records,
            original_bytes=sum(r['original_bytes'] for r in records), bytes=sum(r['bytes'] for r in records),
            scale_alias_bytes=sum(r['original_bytes']-r['bytes'] for r in records),
            source_alias_values_verified=False, compiled=False, physical=False)


def lower_mlp_banks(previous, compact):
    result=deepcopy(previous); result['schema']='compact-mlp-layer-banks-v1'
    result['source_allocated_bytes']=previous['allocated_bytes']
    shifts={pe:r['original_bytes']-r['bytes'] for pe,r in compact.records.items()}
    for p in result['bank_prefixes']:
        pe=tuple(p['pe'])
        if pe in shifts:
            if p['bytes']!=compact.records[pe]['original_bytes']: raise ValueError('MLP prefix provenance')
            p['bytes']-=shifts[pe]
    for p in result['bank_ends']: p['bytes']-=shifts.get(tuple(p['pe']),0)
    for s in result['spans']: s['byte_offset']-=shifts.get(tuple(s['pe']),0)
    result['scale_alias_bytes']=sum(shifts.values())
    result['allocated_bytes']=sum(p['bytes'] for p in result['bank_ends'])
    result['retained_pages_changing_pe']=0
    result['compiled']=False; result['physical']=False; result['source_alias_values_verified']=False
    audit_mlp_banks(previous,compact,result)
    return result


def audit_mlp_banks(previous, compact, candidate):
    if (candidate['schema']!='compact-mlp-layer-banks-v1' or
        candidate['objects']!=previous['objects'] or
        candidate['retained_pages']!=previous['retained_pages'] or
        candidate['conversations']!=1 or candidate['work_slots']!=2 or
        candidate['retained_source_request']!=previous['retained_source_request']):
        raise ValueError('Retained original state/slot contract changed')
    shifts={pe:r['original_bytes']-r['bytes'] for pe,r in compact.records.items()}
    for name,field in [('bank_prefixes','bytes'),('bank_ends','bytes'),('spans','byte_offset')]:
        expected=deepcopy(previous[name])
        for p in expected: p[field]-=shifts.get(tuple(p['pe']),0)
        if candidate[name]!=expected: raise ValueError('Original page/PE identity or prefix shift changed')
    if (candidate['scale_alias_bytes']!=sum(shifts.values()) or
        candidate['allocated_bytes']!=previous['allocated_bytes']-sum(shifts.values()) or
        candidate['allocated_bytes']!=sum(p['bytes'] for p in candidate['bank_ends'])):
        raise ValueError('Compact MLP byte accounting')
    return dict(passed=True, pages=candidate['retained_pages'], scale_alias_bytes=sum(shifts.values()), physical=False)


class CompactMlpAuxiliaryPlacement:
    def __init__(self, previous, compact, candidate):
        audit_mlp_banks(previous,compact,candidate)
        self.spans=candidate['spans'];self.starts=[s['page_start'] for s in self.spans]

    def page(self, page):
        i=bisect_right(self.starts,page)-1
        if i<0 or page>=self.spans[i]['page_start']+self.spans[i]['pages']:
            raise ValueError('Removed or invalid original page')
        s=self.spans[i]
        return dict(pe=s['pe'],byte_offset=s['byte_offset']+128*(page-s['page_start']),bytes=128)


def compact_mlp_csl(files):
    def one(raw,before,after):
        if raw.count(before)!=1: raise ValueError('Original native MLP source anchor changed')
        return raw.replace(before,after)
    raw=files['layer_native.csl']
    raw=one(raw,b'var parts:u16=0;',b'var scale_first:u16=0;\nvar parts:u16=0;')
    raw=one(raw,b'base0:u16,base1:u16) void',b'base0:u16,base1:u16,first_output:u16) void')
    raw=one(raw,b'epoch=new_epoch;parts=k_parts;',b'scale_first=first_output%128;epoch=new_epoch;parts=k_parts;')
    raw=one(raw,b'const offset=base[branch]+iteration*65;@assert(offset+65<=bank_words);',
        b'const offset=base[branch]+iteration*64;const scale_word=base[branch]+iterations*64+((scale_first+iteration*rows)>>7);@assert(offset+64<=bank_words and scale_word<bank_words);')
    raw=one(raw,b'const offset=base[branch]+(iteration*parts+part)*65;@assert(offset+65<=bank_words);',
        b'const offset=base[branch]+(iteration*parts+part)*64;const scale_word=base[branch]+iterations*parts*64+((scale_first+iteration*rows)>>7)*parts+part;@assert(offset+64<=bank_words and scale_word<bank_words);')
    if raw.count(b'bank[offset+64]')!=2: raise ValueError('Original scale read surface changed')
    files['compact_layer_native.csl']=raw.replace(b'bank[offset+64]',b'bank[scale_word]')
    source=one(files['layer_projection.csl'],b'"layer_native.csl"',b'"compact_layer_native.csl"')
    source=one(source,b'setup[2],setup[3]);',b'setup[2],setup[3],setup[4]);')
    files['compact_layer_projection.csl']=source
