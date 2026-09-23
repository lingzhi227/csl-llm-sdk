"""Complete vertical Layer3 roles and accepted retained-bank contracts."""
from collections import Counter

def roles(plan,transport):
    w,h=plan['application'];assert (w,h)==(29,1160)
    result={(x,y):dict(role=0,ordinal=0,participants=54,matrix_kind=0,matrix_group=0,head_id=0) for y in range(h) for x in range(w)}
    def put(point,role,**extra):
        point=tuple(point);assert point in result and result[point]['role']==0;result[point].update(role=role,**extra)
    for name,role in [('origin',6),('input_norm',2),('post_norm',3),('observer',7)]:put(plan[name],role)
    for key,role in [('attention_heads',4),('head_observers',8),('mlp_owners',5)]:
        for i,p in enumerate(plan[key]):put(p,role,head_id=i if role in (4,8) else 0)
    for s in plan['stripes']:
        for i in range(s['columns']):put((s['x'],s['y']+i),1,ordinal=i,participants=s['columns'],matrix_kind=s['packet_kind'] if i==0 else 0,matrix_group=s['group'] if i==0 and s['packet_kind'] in (2,3) else 0)
    for n in transport['nodes']:put((n['x'],n['y']),9,mask=n['mask'],spine=n['spine'])
    assert Counter(p['role'] for p in result.values())=={0:2237,1:30576,2:1,3:1,4:24,5:136,6:1,7:1,8:24,9:639}
    return result

def accepted_arrays(params):
    role = params['role']
    expected = {name:(size,4) for name,size in {
        'packets.status':32,
        'native_status':64, 'credit_status':64}.items()}
    if role == 9:
        result={'config':(4,2),'status':(128,4)}
        for port in range(3):
            if params['mask'] & (1<<port):result[f'p{port}.buffer']=(128,4)
        return result
    endpoint=(role==1 and params['ordinal']==0 and params['matrix_kind'] not in (2,3)) or role in (2,3,4,5,6)
    expected['packets.config']=(4,2)
    if endpoint:
        expected.update({'packets.receive_buffer':(128,4),'packets.transmit_buffer':(128,4)})
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

def expected_arrays(params,exports):
    result=accepted_arrays(params);role=params['role'];root=role==1 and params['ordinal']==0
    for item in exports:
        if role not in item['roles'] and not (root and 'matrix_root' in item['roles']):continue
        name=item.get('bank_by_role',{}).get(str(role),item['bank'])
        count=item.get('count_by_role',{}).get(str(role),item['count']);size=2 if item['dtype']=='u16' else 4
        if name in result:
            assert result[name][0]>=count*size and result[name][1]==size
        else:result[name]=(count*size,size)
    return result
