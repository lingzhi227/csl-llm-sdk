"""Assign unique physical application coordinates; this is not routed CSL.

40/48/136-wide contraction strips preserve local horizontal reduction chains.
Mixed row patterns avoid the 761-column per-layer rounding failure. Vertical
activation distribution and complete operator/control programs remain separate.
"""
import argparse,collections,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def build(reserve_bus=False,bf16_rows=144):
    capacity_file='capacity.json' if bf16_rows==144 else f'capacity-bf16-{bf16_rows}.json'
    capacity=json.loads((ROOT/'configs'/capacity_file).read_text())
    tensors=json.loads((ROOT/'configs/tensors.json').read_text())['tensors']
    queues={w:collections.deque() for w in [40,48,136]};assignments={}
    for name,spec in sorted(capacity['assignments'].items()):
        if spec.get('tile_grid'):
            rows,width=spec['tile_grid'];assert width in queues
            assignments[name]=dict(role=spec['role'],tile=[spec['tile_rows'],128],shape=spec['shape'],strips=[])
            for row in range(rows):queues[width].append((name,row))
    strip_counts={w:len(q) for w,q in queues.items()}
    assert strip_counts=={40:15530 if bf16_rows==144 else 15628,48:1216,136:1216}
    occupied=bytearray(750*1160)
    def claim(x,y,count,role):
        assert 0<=y<1160 and 0<=x<x+count<=750
        begin=y*750+x;assert not any(occupied[begin:begin+count])
        occupied[begin:begin+count]=bytes([role])*count
    remaining=strip_counts[40]-7595;full_rows,tail=divmod(remaining,18)
    assert tail<=17
    patterns=[([136]*4+[40]*5)]*304+[([48]*3+[40]*15)]*405+[([40]*18)]*full_rows+[[48]+[40]*tail]
    for y,pattern in enumerate(patterns):
        x=0
        for width in pattern:
            name,row=queues[width].popleft();claim(x,y,width,1)
            assignments[name]['strips'].append([row,x,y]);x+=width
    assert not any(queues.values()) and len(patterns)<1159
    # These cells never overlap any matrix strip. Reserving them before other
    # weights/state makes a bottom broadcast spine and an east result spine.
    bus=[]
    if reserve_bus:
        bus=[[749,y] for y in range(1160)]+[[x,1159] for x in range(749)]
        for x,y in bus:claim(x,y,1,6)
    free=collections.deque(i for i,v in enumerate(occupied) if not v)
    def cells(count,role):
        ids=[free.popleft() for _ in range(count)]
        for i in ids:claim(i%750,i//750,1,role)
        return [[i%750,i//750] for i in ids]
    for name,spec in capacity['assignments'].items():
        if spec.get('role')=='other':
            count=spec['pe_interval'][1]-spec['pe_interval'][0]
            assignments[name]=dict(role='other',max_tensor_bytes_per_pe=32768,coordinates=cells(count,2))
        elif 'shared_with' in spec:
            assignments[name]=dict(shared_with=spec['shared_with'],scale_index=spec['scale_index'])
    assert set(assignments)==set(capacity['assignments'])
    # Actual short-context state partition: two 128-key x64-value FP32 shards;
    # four 96x64 BF16 K/V shards per full-attention KV head.
    recurrent={};conv={};kv={}
    for layer in range(64):
        if layer%4!=3:
            recurrent[str(layer)]=cells(96,3);conv[str(layer)]=cells(5,4)
        else:kv[str(layer)]=cells(16,5)
    assert sum(v==1 or v==2 for v in occupied)==capacity['weight_pes']
    assert sum(v in [3,4,5] for v in occupied)==5104
    assert len(free)==750*1160-capacity['weight_pes']-5104-len(bus)
    for name,assignment in assignments.items():
        if 'strips' not in assignment:continue
        spec=capacity['assignments'][name];assignment['strips'].sort()
        assert [r for r,x,y in assignment['strips']]==list(range(spec['tile_grid'][0]))
        assert len(assignment['strips'])*spec['tile_grid'][1]==spec['pe_interval'][1]-spec['pe_interval'][0]
        assignment['columns']='tile column k is at (strip.x+k, strip.y)'
        assignment['row_extent']='min(tile_rows, original_rows - tile_row*tile_rows); zero pad remaining local rows'
    return dict(model=capacity['repository'],revision=capacity['revision'],application=[750,1160],physical_fabric=[762,1172],fabric_offset=[4,1],
        status='Collision-free storage/state coordinates only; no complete routed or compiled model',context=96,
        matrix_pes=sum(v==1 for v in occupied),other_weight_pes=sum(v==2 for v in occupied),state_pes=sum(v in [3,4,5] for v in occupied),
        weight_strip_rows=len(patterns),bf16_tile_rows=bf16_rows,strip_counts=strip_counts,unallocated_control_and_route_pes=len(free),assignments=assignments,
        states=dict(recurrent=recurrent,conv=conv,kv=kv),
        recurrent_partition=dict(keys=128,values_per_shard=64,shards_per_head=2,state_dtype='FP32'),
        bus=dict(reserved=reserve_bus,coordinates=bus,pes=len(bus),status='Route architecture only; not compiled as a whole model'),
        control_pool=[[i%750,i//750] for i in free],
        unresolved=['Vertical activation multicast across interleaved matrix strips','BF16 and embedding communication program SRAM',
            'All role placement/code/queue/color/routing compilation','Operator/control programs in remaining pool',
            'Full vocabulary and all-layer continuous device generation'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reserve-bus',action='store_true');parser.add_argument('--bf16-rows',type=int,choices=[144,140],default=144);args=parser.parse_args()
    result=build(args.reserve_bus,args.bf16_rows)
    filename='placement-resident-96.json' if args.reserve_bus else 'placement-96.json'
    if args.bf16_rows!=144:filename=filename.replace('.json',f'-bf16-{args.bf16_rows}.json')
    (ROOT/'configs'/filename).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['assignments','states','control_pool','bus']},indent=2))
