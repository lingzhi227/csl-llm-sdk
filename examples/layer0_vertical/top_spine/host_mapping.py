"""Plan-derived legal ROI transfers for vertical weights, state and boundaries."""
from pathlib import Path
from collections import defaultdict
import json
HERE=Path(__file__).resolve().parent
MAX_HOST_BYTES=16<<20

def rectangles(points,count):
 """Merge equal horizontal runs, then bound their vertical host-slot size."""
 rows=defaultdict(list);runs=defaultdict(list)
 for x,y in points:rows[y].append(x)
 for y,xs in sorted(rows.items()):
  start=last=None
  for x in sorted(xs):
   if start is not None and x!=last+1:runs[start,last-start+1].append(y);start=None
   if start is None:start=x
   last=x
  if start is not None:runs[start,last-start+1].append(y)
 result=[]
 for (x,width),ys in sorted(runs.items()):
  limit=MAX_HOST_BYTES//(width*count*4);assert limit>=1
  first=last=None
  for y in ys:
   if first is not None and (y!=last+1 or y-first>=limit):result.append([x,first,width,last-first+1,count]);first=None
   if first is None:first=y
   last=y
  if first is not None:result.append([x,first,width,last-first+1,count])
 return sorted(result,key=lambda b:(b[1],b[0]))
def transfer(symbol,dtype,box,**extra):
 x,y,w,h,n=box;native=2 if dtype=='u16' else 4
 return dict(symbol=symbol,dtype=dtype,x=x,y=y,width=w,height=h,count=n,bits=native*8,
  order='ROW_MAJOR',host_shape=[h,w,n],host_bytes=w*h*n*4,native_bytes=w*h*n*native,**extra)
def build(plan,transport):
 width,height=plan['application'];matrix={};uploads=[];config=[]
 for i,s in enumerate(plan['stripes']):
  assert s['physical_step']==[0,1] and s['axis']=='y'
  for ordinal in range(s['columns']):
   p=(s['x'],s['y']+ordinal);assert p not in matrix;matrix[p]=(i,ordinal)
 assert len(matrix)==31548
 for i,box in enumerate(rectangles(matrix,6144)):
  x,y,w,h,n=box;groups=defaultdict(list)
  for yy in range(y,y+h):
   for xx in range(x,x+w):logical,ordinal=matrix[xx,yy];groups[logical].append((ordinal,xx-x,yy-y))
  segments=[]
  for logical,points in sorted(groups.items()):
   points.sort();first,dx,dy=points[0]
   assert points==[(first+j,dx,dy+j) for j in range(len(points))]
   segments.append(dict(logical_stripe=logical,original_file=f'weights-{logical:03}.npy',ordinal_begin=first,
     tile_count=len(points),destination_x=dx,destination_y=dy))
  uploads.append(transfer('weights','u32',box,file=f'weight-roi-{i:03}.npy',segments=segments,
    original_source='Accepted preparation002 packed u32 words; no original tensor transpose or model reread'))
 assert sum(u['host_bytes'] for u in uploads)==775323648
 config.append(transfer('transport_config','u16',[0,0,width,height,2],file='transport-config.npy'))
 for symbol,count in [('descriptor',6),('route',4)]:
  for box in rectangles(matrix,count):config.append(transfer(symbol,'u16',box,file=symbol+'.npy',file_x=box[0],file_y=box[1]))
 owners=[]
 for i,c in enumerate(plan['convolution']):owners.append((c['owner'],i))
 for i,h in enumerate(plan['heads']):
  owners.append((h['owner'],i))
  owners.extend((s['owner'],4*i+j) for j,s in enumerate(h['state_shards']))
 owners.extend((xy,i) for i,xy in enumerate(plan['support']['MLP_owners']))
 for box in rectangles({tuple(p) for p,i in owners},4):config.append(transfer('route','u16',box,file='route.npy',file_x=box[0],file_y=box[1]))
 parameters=[]
 for name,file in [('input_norm','input-norm.npy'),('post_norm','post-norm.npy')]:
  x,y=plan['support'][name];parameters.append(transfer('norm_gains','u16',[x,y,1,1,5120],file=file,file_row=0))
 def ordered(symbol,dtype,points,count,file=None,**extra):
  x,y=points[0];assert points==[[x,y+i] for i in range(len(points))]
  return transfer(symbol,dtype,[x,y,1,len(points),count],**(dict(file=file,file_row=0) if file else {}),**extra)
 conv=[c['owner'] for c in plan['convolution']];heads=[h['owner'] for h in plan['heads']]
 parameters.extend([ordered('conv_weights','u16',conv,512,'convolution-weights.npy'),
  ordered('head_gains','u16',heads,128,'head-gains.npy'),ordered('head_parameters','u16',heads,2,'head-parameters.npy')])
 shards=[s['owner'] for h in plan['heads'] for s in h['state_shards']]
 persistent=[ordered('conv_history','u16',conv,512,canonical_shape=[10240,4],canonical_order='channel,tap'),
  ordered('recurrent_matrix','f32',shards,4096,canonical_shape=[48,128,128],physical_order='head,value_shard,key,local_value',canonical_order='head,key,value')]
 hidden={name:transfer('hidden','u16',[*plan['support'][role],1,1,5120]) for name,role in [('hidden_input','input_norm'),('hidden_output','post_norm')]}
 counts=dict(weight_H2D=dict(calls=len(uploads),host_bytes=sum(x['host_bytes'] for x in uploads)),
  original_parameter_H2D=dict(calls=len(parameters),host_bytes=sum(x['host_bytes'] for x in parameters)),
  configuration_H2D=dict(calls=len(config),host_bytes=sum(x['host_bytes'] for x in config)),
  checkpoint_payload_D2H=dict(calls=len(persistent),host_bytes=sum(x['host_bytes'] for x in persistent),native_bytes=sum(x['native_bytes'] for x in persistent)),
  restore_payload_H2D=dict(calls=len(persistent),host_bytes=sum(x['host_bytes'] for x in persistent),native_bytes=sum(x['native_bytes'] for x in persistent)),
  stage_hidden_each_direction=dict(calls=1,host_bytes=20480,native_bytes=10240))
 return dict(scope='Generated legal export-role ROI intent; actual compiler banks and runtime must still qualify it',application=plan['application'],
  transfer_contract=dict(channels=16,max_channels_in_installed_frontend=16,channel_height_requirement_passed=16<=height,
   order='ROW_MAJOR',host_slots_bits=32,maximum_host_copy_bytes=MAX_HOST_BYTES,active_host_calls=1,nonblock=False,
   physical_x_unchanged=True,channel_choice='Use all16 supported DMA channels for the long1160-row ROI; one blocking RPC may span multiple compiled channels. No measured speedup claim.'),
  matrix_uploads=uploads,config_uploads=config,parameter_uploads=parameters,persistent_exports=persistent,**hidden,budget=counts,
  checkpoint_manifest=dict(required=['model_revision','layer_ids','request_id','reset_generation','next_position','weight_manifest_sha256','state_payload_hashes','quiescent_completion_receipt'],
   payload_banks=['conv_history','recurrent_matrix'],KV='No KV cache in linear-attention Layer0; Layer3 attention must add its complete K/V state.',
   control_restore_implemented=False,restore_note='H2D payload groups alone do not restore private wrapper/protocol position. A separately qualified restore command and checkpoint identity are required in stage integration.'),
  measurement_fields=['compiled_channels','direction','copy_category','rectangle','host_bytes','native_bytes','submission_seconds','RPC_return_seconds','raw_fsync_seconds','SDK_startup_seconds','stop_seconds'],
  actual_compiled_banks_bound=False,SDK_invocations=0,measured_performance=False)
def configuration_arrays(plan,transport):
 """Small remote configuration generator; original parameter values stay separate."""
 import numpy as np
 w,h=plan['application'];config=np.zeros((h,w,2),dtype='<u2');descriptor=np.zeros((h,w,6),dtype='<u2');route=np.zeros((h,w,4),dtype='<u2')
 for y in range(h):config[y,:,0]=np.arange(w,dtype='<u2');config[y,:,1]=y
 for e in transport['endpoints']:
  x,y=e['endpoint'];config[y,x]=e['router']
 for s in plan['stripes']:
  for ordinal in range(s['columns']):
   x,y=s['x'],s['y']+ordinal
   descriptor[y,x]=[s['phase'],s['group'],s['legacy_consumer_label'],0,s['valid_rows'],s['valid_last_columns'] if ordinal==s['columns']-1 else 96]
   route[y,x]=[s['kind'],len(s['targets']),0,0]
 groups=[[(c['owner'],i) for i,c in enumerate(plan['convolution'])],[(h['owner'],i) for i,h in enumerate(plan['heads'])],[(s['owner'],4*i+j) for i,h in enumerate(plan['heads']) for j,s in enumerate(h['state_shards'])],[(xy,i) for i,xy in enumerate(plan['support']['MLP_owners'])]]
 for group in groups:
  for (x,y),index in group:route[y,x,2]=index
 return {'transport-config.npy':config,'descriptor.npy':descriptor,'route.npy':route}
if __name__=='__main__':
 value=build(json.loads((HERE/'plan.json').read_bytes()),json.loads((HERE/'transport-map.json').read_bytes()))
 (HERE/'host-map.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value['budget']))
