"""Logical views reconstructed exclusively from actual physical raw receipts."""
import numpy as np
from archive_decode import decode_archive
from runtime_boundary import require
class Views:
 def __init__(self,evidence,prepared,physical,copies,logical):
  self.e=evidence;self.prepared=prepared;self.physical=physical;self.logical=logical
  self.stripes={(s['name'],s['group']):s for s in physical['stripes']};self.weights_by_tile={}
  for item in copies['uploads']:
   if item['symbol']!='weights':continue
   source=item['prepared'];width=source['width']
   for segment in source['segments']:
    stripe=physical['stripes'][segment['logical_stripe']]
    for i in range(segment['tile_count']):
     key=(stripe['name'],stripe['group'],segment['ordinal_begin']+i)
     require(key not in self.weights_by_tile,'One original packed tile')
     self.weights_by_tile[key]=(source['file'],source['file_flat_begin']+(segment['destination_y']+i)*width+segment['destination_x'])
  require(len(self.weights_by_tile)==30576,'Every original matrix tile')
 def matrix(self,serial,index):
  b=self.logical['weight_batches'][index];result={}
  for name,count in [('input',96),('active_input',96),('partial',128),('result',128),('lane_counters',4),('rounded',128)]:
   values=[]
   for group in b['row_groups']:
    s=self.stripes[b['name'],group]
    require(s['columns']==b['width'],'Original logical input ordinal count')
    value=self.e.rectangle(serial,name,s['x'],s['y'],1,1 if name=='rounded' else s['columns'],count)
    values.append(value[:,0])
   result[name]=np.stack(values)
  return result
 def weight_batch(self,index):
  b=self.logical['weight_batches'][index];result=np.empty((b['height'],b['width'],6144),dtype='<u4');by_file={}
  for y,g in enumerate(b['row_groups']):
   for x in range(b['width']):
    filename,row=self.weights_by_tile[b['name'],g,x];by_file.setdefault(filename,[]).append((y,x,row))
  for name,points in by_file.items():
   source,_=self.prepared.array(name)
   for y,x,row in points:result[y,x]=source[row]
  return result
 def physical_coordinate(self,batch,row,ordinal):
  b=self.logical['weight_batches'][batch];s=self.stripes[b['name'],b['row_groups'][row]]
  return s['x'],s['y']+ordinal
 def owners(self,serial,name,points):return np.stack([self.e.coordinate(serial,name,*p) for p in points])
 def peers(self,serial):
  result={};p=self.physical
  for name in ['hidden','norm_saved','norm_stats']:result[name]=self.owners(serial,name,[p['input_norm'],p['post_norm']])
  for name in ['qk_input','qk_output','qk_stats','attention_input','attention_scores','attention_score_casts','attention_stats','attention_stages','attention_casts','cache']:
   result[name]=self.owners(serial,name,p['attention_heads'])
  for name in ['mlp_gate','mlp_up','mlp_product','mlp_silu','mlp_exponential','mlp_sigmoid','mlp_activation','mlp_product_fp32']:
   result[name]=self.owners(serial,name,p['mlp_owners'])
  result['frame_counters']=self.e.coordinate(serial,'frame_counters',*p['origin'])
  return result
 def archives(self,serial,reset,position):
  points=self.physical['head_observers'];status=self.owners(serial,'qk_archive_status',points);data=self.owners(serial,'qk_archive',points)
  result=[decode_archive(status[h].tolist(),data[h].tolist(),head=h,identity=[1,reset,position,0,0],generation=serial) for h in range(24)]
  for h,v in enumerate(result):v['sink']=list(points[h])
  return result
