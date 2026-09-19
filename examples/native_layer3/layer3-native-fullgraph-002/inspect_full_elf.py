"""Actual complete native graph placement and original storage at all33750 PEs."""
import hashlib
import json
from pathlib import Path
from cerebras.elf.cself import ELFMemory
from elf_symbols import symbols
from hw00.store import atomic_json

ROOT = Path(__file__).resolve().parent
from fullgraph_roles import roles
ROLES = roles(json.loads((ROOT/'layer3-plan.json').read_bytes()))
ALL = {(x + 4, y + 1) for y in range(45) for x in range(750)}
SRAM = json.loads((ROOT / 'sram.json').read_bytes())['application_files']


def array(records, name, size, alignment=4):
    found = {}
    for item in records:
        canonical = item['name'].rsplit('$$', 1)[-1] if item['name'].startswith('$$csl_base_address$$') else item['name']
        if (canonical == name or size == 4 and canonical == name + '.0') and item['bytes'] > 0:
            found[item['address'], item['bytes']] = item
    if len(found) != 1:
        raise ValueError('One actual array required: ' + name)
    value = next(iter(found.values()))
    if value['bytes'] != size or value['address'] % alignment:
        raise ValueError('Actual extent/alignment: ' + name)
    return value


def expected_arrays(params):
    role = params['role']
    expected = {name:(size,4) for name,size in {
        'packets.header_buffer':4, 'packets.receive_buffer':124,
        'native_status':64, 'credit_status':64}.items()}
    if role == 1:
        expected.update({name:(size,4) for name,size in {
            'lane.weight_storage':24576, 'lane.input':384, 'lane.active_input':384,
            'lane.kernel.expanded':512, 'lane.partial':512, 'lane.result':512}.items()})
        if params['ordinal'] == 0:
            expected['lane.rounded'] = 256, 2
    if role in (2,3):
        expected.update({name:(10240,2) for name in ('norm.gains','norm.inout','norm.saved')})
    if role == 4:
        expected.update({name:(size,2) for name,size in {
            'attn.cache':8192, 'attn.input':2048, 'attn.qk_input':1024,
            'attn.qk_weights':1024, 'attn.casts':1536}.items()})
        expected.update({name:(size,4) for name,size in {
            'attn.stages':5120, 'attn.qk_stats':40,
            'head_observer.buffer':128}.items()})
    if role == 5:
        expected.update({name:(256,2) for name in ('product.gate','product.up','product.product','product.silu')})
        expected.update({name:(512,4) for name in ('product.exponential','product.sigmoid','product.activation_fp32','product.product_fp32')})
    if role in (6,7):
        expected['observer.buffer'] = 128,4
    if role == 6:
        expected['origin_credit.states'] = 192,2
        expected['origin_fault.words'] = 44,4
    if role == 7:
        expected['observer.snapshot'] = 128,4
    if role == 8:
        expected['head_observer.buffer'] = 128,4
        expected['head_observer.snapshot'] = 128,4
        expected['head_observer.archive_storage'] = 3840,4
        expected['head_observer.archive.status'] = 64,4
    return expected


def main():
    covered, rows = set(), []
    paths = sorted((ROOT / 'compiled/out/bin').glob('*.elf'))
    if not 1 <= len(paths) <= 1536 or {p.name for p in paths} != set(SRAM):
        raise ValueError('Bounded actual ELF inventory')
    for path in paths:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SRAM[path.name]['sha256']:
            raise ValueError('Actual inspected ELF identity')
        reader = ELFMemory(path)
        if tuple(reader.get_fabric_dimensions()) != (762,1172):
            raise ValueError('Original full fabric geometry')
        rectangles = set()
        count = 0
        for rectangle in reader.iter_rectangles():
            count += 1
            if count > 33750 * 32:
                raise ValueError('Raw placement metadata bound')
            rectangles.add(tuple(rectangle))
        coordinates = set()
        for x,y,w,h in rectangles:
            if w < 1 or h < 1 or x < 4 or y < 1 or x+w > 754 or y+h > 46:
                raise ValueError('Rectangle outside original application')
            coordinates.update((xx,yy) for xx in range(x,x+w) for yy in range(y,y+h))
        if not coordinates or not coordinates <= ALL or coordinates & covered:
            raise ValueError('Disjoint actual program placement')
        covered |= coordinates
        records = symbols(raw)
        # One ELF can serve many PEs; validate every distinct storage contract.
        contracts={tuple(sorted(expected_arrays(ROLES[point]).items())) for point in coordinates}
        if len(contracts)!=1:
            raise ValueError('One actual program merged incompatible original storage roles')
        arrays={name:array(records,name,size,alignment)
                for name,(size,alignment) in next(iter(contracts))}
        role_ids={ROLES[point]['role'] for point in coordinates}
        if len(role_ids)!=1:raise ValueError('One program merged different fullgraph roles')
        role=next(iter(role_ids))
        alias_views={}
        if role==4:
            if len(coordinates)!=1:raise ValueError('Distinct original head identity')
            relocated=('normalized','trig','rotary_stages','product_casts','trig_casts','probes')
            for item in records:
                canonical=item['name'].rsplit('$$',1)[-1] if item['name'].startswith('$$csl_base_address$$') else item['name']
                if canonical in {'attn.'+name for name in relocated} and item['type']==1 and item['bytes']>0:
                    raise ValueError('Obsolete independent head diagnostic bank')
                if any(('attn.'+name+'_storage') in canonical for name in relocated) and item['bytes']>4:
                    raise ValueError('Unexpected retained fallback diagnostic bank')
            base=arrays['attn.stages']['address']
            for name,start,end in [('trig',0,512),('rotary_stages',512,2048),
                    ('product_casts',2048,2560),('trig_casts',2560,2688),
                    ('probes',2688,2816),('normalized',4096,5120)]:
                alias_views[name]=dict(address=base+start,bytes=end-start,
                    evidence='Actual workspace base plus pinned source accessor offset; not a separate ELF symbol')
        if role==8:
            if len(coordinates)!=1 or any(ROLES[p]['head_id']!=25-p[1] for p in coordinates):
                raise ValueError('Each actual archive sink binds head24-application_y')
        if role==0 and any('weight_storage' in v['name'] and v['bytes']==24576 for v in records):
            raise ValueError('Unexpected neural weight bank in original idle role')
        values = list(arrays.values())
        for index,left in enumerate(values):
            for right in values[index+1:]:
                if max(left['address'],right['address']) < min(left['address']+left['bytes'],right['address']+right['bytes']):
                    raise ValueError('Live retained/source/packet arrays overlap: '+left['name']+' '+right['name'])
        rows.append(dict(file=path.name,rectangles=sorted(rectangles),role=role,
            PEs=len(coordinates),arrays=arrays,source_derived_alias_views=alias_views))
    if covered!=ALL or covered!=set(ROLES):
        raise ValueError('Every original33750 PE must be covered exactly once')
    atomic_json(ROOT/'placement.json',dict(passed=True,PEs=len(covered),matrix_PEs=30576,
        full_heads=24,native_sources=16,archive_sinks=24,MLP_owners=136,
        application=[750,45],geometry=[762,1172],files=rows,runtime_created=False,
        executable_graph=True,demoted_PEs=0,passed_scope='fullgraph_placement_and_original_storage_only',
        sram_gate_separate=True,
        alias_scope='Actual workspace/bank extents plus pinned source offsets and separately accepted physical fixture',
        scope='Complete native layer3 graph compile only; no runtime, dynamic-stack or neural acceptance'))



if __name__ == '__main__':
    main()
