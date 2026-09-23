"""Legal vertical ROIs and exact accepted original packed-word provenance."""
from collections import defaultdict
from pathlib import Path
import json
from roi import rectangles, transfer

HERE=Path(__file__).resolve().parent

def build(plan,transport,original):
    w,h=plan['application'];matrix={};source={};uploads=[];config=[]
    for i,batch in enumerate(original['weight_batches']):
        for row,group in enumerate(batch['row_groups']):
            key=(batch['name'],group);assert key not in source
            source[key]=dict(file=f'weights-{i:03}.npy',row=row,width=batch['width'])
    for logical,s in enumerate(plan['stripes']):
        assert s['axis']=='y' and s['physical_step']==[0,1]
        item=source[s['name'],s['group']];assert item['width']==s['columns']
        for ordinal in range(s['columns']):
            xy=(s['x'],s['y']+ordinal);assert xy not in matrix;matrix[xy]=(logical,ordinal)
    assert len(matrix)==30576
    for i,box in enumerate(rectangles(matrix,6144)):
        x,y,width,height,count=box;groups=defaultdict(list)
        for yy in range(y,y+height):
            for xx in range(x,x+width):
                logical,ordinal=matrix[xx,yy];groups[logical].append((ordinal,xx-x,yy-y))
        segments=[]
        for logical,points in sorted(groups.items()):
            points.sort();first,dx,dy=points[0]
            assert points==[(first+j,dx,dy+j) for j in range(len(points))]
            s=plan['stripes'][logical];old=source[s['name'],s['group']]
            segments.append(dict(logical_stripe=logical,original_file=old['file'],original_row=old['row'],
                ordinal_begin=first,tile_count=len(points),destination_x=dx,destination_y=dy))
        uploads.append(transfer('weights','u32',box,file=f'weight-roi-{i:03}.npy',segments=segments))
    assert sum(u['host_bytes'] for u in uploads)==751435776
    config.append(transfer('transport_config','u16',[0,0,w,h,2],file='transport-config.npy'))
    for symbol,count in [('descriptor',6),('route',4)]:
        for box in rectangles(matrix,count):config.append(transfer(symbol,'u16',box,file=symbol+'.npy',file_x=box[0],file_y=box[1]))
    for box in rectangles({tuple(p) for p in plan['attention_heads']+plan['mlp_owners']},4):
        config.append(transfer('route','u16',box,file='route.npy',file_x=box[0],file_y=box[1]))
    parameters=[]
    for key,file in [('input_norm','input-norm.npy'),('post_norm','post-norm.npy')]:
        parameters.append(transfer('norm_gains','u16',[*plan[key],1,1,5120],file=file,original_row=None))
    for head,point in enumerate(plan['attention_heads']):
        for symbol,dtype,count,file in [('qk_gains','u16',512,'qk-norms.npy'),('frequencies','f32',32,'frequencies.npy')]:
            parameters.append(transfer(symbol,dtype,[*point,1,1,count],file=file,original_row=head))
    persistent=[transfer('cache','u16',[*point,1,1,4096],head=head,
        physical_shape=[2,8,256],order_within_PE='key_or_value,position,dimension') for head,point in enumerate(plan['attention_heads'])]
    hidden={label:transfer('hidden','u16',[*plan[key],1,1,5120]) for label,key in [('hidden_input','input_norm'),('hidden_output','post_norm')]}
    return dict(scope='Generated role-exact copy intent; actual compiled banks and original preparation must still qualify it',
        application=[w,h],matrix_uploads=uploads,config_uploads=config,parameter_uploads=parameters,
        persistent_exports=persistent,**hidden,
        transfer_contract=dict(channels=16,physical_x_unchanged=True,order='ROW_MAJOR',host_slots_bits=32,maximum_host_copy_bytes=16<<20),
        budget=dict(weight_H2D=dict(calls=len(uploads),host_bytes=sum(u['host_bytes'] for u in uploads)),
            parameter_H2D=dict(calls=len(parameters),host_bytes=sum(u['host_bytes'] for u in parameters)),
            configuration_H2D=dict(calls=len(config),host_bytes=sum(u['host_bytes'] for u in config)),
            full_physical_KV_each_direction=dict(calls=24,host_bytes=24*4096*4,native_bytes=24*4096*2)),
        checkpoint_contract=dict(full_physical_KV_shape=[24,2,8,256],grouped_key_value_heads=4,query_heads_per_KV_head=6,
            duplicated_KV_state_preserved=True,control_restore_implemented=False,
            required_identity=['model_revision','layer_id','request_id','reset_generation','next_position','weight_manifest_sha256','state_payload_hashes','quiescent_completion_receipt']),
        actual_compiled_banks_bound=False,SDK_invocations=0,full_layer_accepted=False)

def configuration_arrays(plan,transport):
    """Remote-only metadata arrays; original weights/parameters stay separate."""
    import numpy as np
    w,h=plan['application'];config=np.zeros((h,w,2),dtype='<u2');descriptor=np.zeros((h,w,6),dtype='<u2');route=np.zeros((h,w,4),dtype='<u2')
    for y in range(h):config[y,:,0]=np.arange(w,dtype='<u2');config[y,:,1]=y
    for e in transport['endpoints']:
        x,y=e['endpoint'];config[y,x]=e['router']
    for s in plan['stripes']:
        for ordinal in range(s['columns']):
            x,y=s['x'],s['y']+ordinal
            descriptor[y,x]=[s['phase'],s['group'],s['legacy_consumer'][0],s['legacy_consumer'][1],s['rows'],s['last_columns'] if ordinal==s['columns']-1 else 96]
            route[y,x]=[s['packet_kind'],s['fanout'],0,0]
    for points in (plan['attention_heads'],plan['mlp_owners']):
        for i,(x,y) in enumerate(points):route[y,x,2]=i
    return {'transport-config.npy':config,'descriptor.npy':descriptor,'route.npy':route}

def packed_ROI(item,original_loader):
    """Remote-only exact u32 copy; no BF16 conversion or tensor transpose."""
    import numpy as np
    result=np.zeros(item['host_shape'],dtype='<u4');covered=np.zeros(item['host_shape'][:2],dtype=bool)
    for segment in item['segments']:
        source=original_loader(segment['original_file']);row=segment['original_row'];start=segment['ordinal_begin'];n=segment['tile_count']
        assert source.dtype==np.dtype('<u4') and source.ndim==3 and source.shape[2]==6144
        x,y=segment['destination_x'],segment['destination_y'];assert not covered[y:y+n,x].any()
        result[y:y+n,x]=source[row,start:start+n];covered[y:y+n,x]=True
    assert covered.all() and result.nbytes==item['host_bytes']<=16<<20
    return result

if __name__=='__main__':
    value=build(*[json.loads((HERE/name).read_bytes()) for name in ('plan.json','transport-map.json','logical-plan.json')])
    (HERE/'host-map.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value['budget']))
