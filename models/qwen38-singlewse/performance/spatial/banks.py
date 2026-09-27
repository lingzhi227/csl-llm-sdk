"""Compact cyclic weight-bank plan: all original matrices, small tiles per phase.

This is a storage/ownership plan, not a route map or compiled SRAM admission.
No checkpoint values or per-tile giant tables are materialized on the Mac.
"""
from collections import Counter
from dataclasses import asdict,dataclass
import hashlib,json,math
from pathlib import Path

@dataclass(frozen=True)
class BankPolicy:
    application_pes:int=870000
    reserved_actor_pes:int=16384
    rows_per_tile:int=2
    columns_per_tile:int=128
    descriptor_bytes_per_tile:int=4
    code_allowance_bytes:int=6144
    stack_allowance_bytes:int=4096
    scratch_allowance_bytes:int=1536
    sram_ceiling:int=48128

    def validate(self):
        if not (self.application_pes==870000 and 0<=self.reserved_actor_pes<self.application_pes):raise ValueError('PE policy')
        if self.rows_per_tile not in (2,4,8,16) or self.columns_per_tile!=128:raise ValueError('Scale-aligned tile policy')
        if min(self.descriptor_bytes_per_tile,self.code_allowance_bytes,self.stack_allowance_bytes,self.scratch_allowance_bytes)<0:raise ValueError('Negative resource allowance')
        if self.sram_ceiling!=48128:raise ValueError('SRAM ceiling cannot be raised')


def local_count(total,pe,count,phase=0):
    """Number of tiles owned by one PE after a prefix of the class stream."""
    if not (count>0 and 0<=pe<count and total>=0):raise ValueError('Owner domain')
    return total//count + int((pe-phase)%count < total%count)


def tile_owner(start,tile,count,phase=0):
    if min(start,tile)<0 or count<=0:raise ValueError('Tile domain')
    return (phase+start+tile)%count


def build(root,policy=BankPolicy()):
    policy.validate();root=Path(root)
    paths=[root/'configs'/x for x in ['model-graph.json','tensors.json']]
    graph,header=[json.loads(p.read_text()) for p in paths]
    if graph['revision']!='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a' or header['revision']!=graph['revision']:raise ValueError('Model revision')
    specs=header['tensors'];names=list(dict.fromkeys(n for op in graph['nodes'] for n in op['weights']))
    if len(names)!=1251:raise ValueError('Full original text graph required')
    count=policy.application_pes-policy.reserved_actor_pes
    classes={'fp8':dict(tile_bytes=policy.rows_per_tile*128+4,phase=0,total_tiles=0),
             'bf16':dict(tile_bytes=policy.rows_per_tile*128*2,phase=count//2,total_tiles=0)}
    scale_names={n+'_scale_inv' for n in names if specs[n]['dtype']=='F8_E4M3'}
    records=[];other=[];used=set();padding=0
    for n in names:
        if n in scale_names:continue
        s=specs[n]
        if len(s['shape'])!=2:
            other.append(n);used.add(n);continue
        m,k=s['shape']
        if s['dtype'] not in ('BF16','F8_E4M3') or k%128 or m%policy.rows_per_tile:raise ValueError('Unsupported exact matrix partition: '+n)
        kind='fp8' if s['dtype']=='F8_E4M3' else 'bf16';c=classes[kind]
        tiles=(m//policy.rows_per_tile)*(k//128);start=c['total_tiles'];c['total_tiles']+=tiles
        used.add(n)
        if kind=='fp8':
            sn=n+'_scale_inv';used.add(sn)
            if specs[sn]['dtype']!='BF16' or specs[sn]['shape']!=[math.ceil(m/128),k//128]:raise ValueError('Original scale coverage')
        records.append(dict(tensor=n,kind=kind,shape=[m,k],row_tiles=m//policy.rows_per_tile,k_blocks=k//128,
            global_tile_start=start,tiles=tiles,phase=c['phase'],active_pes=min(count,tiles),
            max_tiles_per_pe=math.ceil(tiles/count),max_weight_macs_per_active_pe=math.ceil(tiles/count)*policy.rows_per_tile*128,
            tile_order='row tile major, then K block',weight_transform='original FP8 bytes + exact FP32 expansion of one original BF16 scale' if kind=='fp8' else 'original BF16 words'))
    if used!=set(names) or len(records)!=498:raise ValueError('Lost or duplicated tensor binding')
    # Only O(PEs) lightweight counts; not a materialization of ~100M model tiles.
    histogram=Counter();maximum=0;minimum=1<<30;overflow=0;sum_payload=0;max_tiles=0
    for pe in range(count):
        counts={kind:local_count(c['total_tiles'],pe,count,c['phase']) for kind,c in classes.items()}
        tile_count=sum(counts.values());payload=sum(counts[kind]*classes[kind]['tile_bytes'] for kind in classes)
        total=payload+tile_count*policy.descriptor_bytes_per_tile+policy.code_allowance_bytes+policy.stack_allowance_bytes+policy.scratch_allowance_bytes
        histogram[total]+=1;maximum=max(maximum,total);minimum=min(minimum,total);overflow+=total>policy.sram_ceiling;sum_payload+=payload;max_tiles=max(max_tiles,tile_count)
    original_bytes=sum(specs[n]['bytes'] for n in names)
    if original_bytes!=29468003328:raise ValueError('Original payload identity')
    other_bytes=sum(specs[n]['bytes'] for n in other)
    expected_payload=sum(c['total_tiles']*c['tile_bytes'] for c in classes.values())
    if expected_payload!=sum_payload:raise ValueError('Bank conservation failure')
    return dict(schema='wse-weight-banks-v1',model=graph['model'],revision=graph['revision'],policy=asdict(policy),
        weight_bank_pes=count,classes=classes,matrices=records,reserved_actor_weights=other,
        metrics=dict(original_tensors=len(names),matrices=len(records),original_payload_bytes=original_bytes,
            resident_matrix_bytes_including_replicated_scales=expected_payload,reserved_actor_weight_bytes=other_bytes,
            min_estimated_pe_bytes=minimum,max_estimated_pe_bytes=maximum,max_local_tiles=max_tiles,overflow_pes=overflow,
            estimated_weight_banks_fit=overflow==0,weight_bank_histogram={str(k):v for k,v in sorted(histogram.items())}),
        ownership_formula='pe=(class.phase+matrix.global_tile_start+row_tile*k_blocks+k_block)%weight_bank_pes',
        local_offset_formula='class_base_on_pe + prefix_tile_count_on_pe * class.tile_bytes; metadata stored separately',
        provenance={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        physical_routes_admitted=False,compiled_sram_admitted=False,full_model_executable=False,
        unresolved=['Map compact bank IDs to reserved actor geometry','Verify descriptors/code/stack with compiled ELF and actual access ranges',
            'Place all non-matrix weights/state within reserved actors','Route operand multicast and reductions for every matrix',
            'Prove prefetch-buffer lifetime and measure overlap','Integrate and qualify complete dependent sentence generation'])
