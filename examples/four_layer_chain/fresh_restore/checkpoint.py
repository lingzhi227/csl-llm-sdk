"""Durable actual-device checkpoint manifest and immutable native restore views.

The runtime supplies already completed D2H receipts after its full command,
transport and boundary fence. This module neither creates a runtime nor computes
model state, and cannot replace the independent physical-release admission.
"""
from pathlib import Path
import hashlib,json,os,stat

DTYPES={'u16':'<u2','u32':'<u4','f32':'<f4'}

def identity(item):
 return item['symbol'],item['dtype'],*(item[k] for k in ('x','y','width','height','count'))

def _json_bytes(value):return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def _stamp(path):
 s=path.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode

def _payload(root,item,receipt,require_readonly):
 import numpy as np
 relative=Path(receipt['path'])
 if relative.is_absolute() or '..' in relative.parts or len(relative.parts)!=2 or relative.parts[0]!='raw':raise ValueError('Bounded raw capture path')
 path=root/relative
 if path.is_symlink() or path.parent.is_symlink():raise ValueError('No checkpoint symlink')
 before=_stamp(path)
 if not stat.S_ISREG(before[5]) or require_readonly and before[5]&0o222:raise ValueError('Immutable regular checkpoint payload')
 shape=(item['height'],item['width'],item['count']);dtype=np.dtype(DTYPES[item['dtype']]);size=dtype.itemsize*shape[0]*shape[1]*shape[2]
 if size!=receipt['bytes'] or size!=before[2] or not 0<size<=16<<20:raise ValueError('Exact checkpoint payload extent')
 box=[item[k] for k in ('x','y','width','height','count')]
 if (receipt['symbol'],receipt['dtype'],receipt['rectangle'])!=(item['symbol'],item['dtype'],box):raise ValueError('Raw receipt geometry')
 raw=path.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=receipt['sha256'] or _stamp(path)!=before:raise ValueError('Checkpoint hash/stamp changed')
 value=np.frombuffer(raw,dtype=dtype).reshape(shape)
 if value.flags.writeable or not value.flags.c_contiguous:raise ValueError('Independent immutable native view')
 return value

def control(value,regions,request_id,stage_id,generation=1,next_position=1):
 import numpy as np
 if value.dtype!=np.dtype('<u4') or value.shape!=(1160,119,8):raise ValueError('Full control shape/dtype')
 for r in regions:
  expected=np.array([0x51435031,request_id,stage_id,r['layer'],generation,next_position,1,0],dtype='<u4')
  if np.any(value[:,r['x']:r['x']+r['width'],:]!=expected):raise ValueError('Every PE logical checkpoint identity')

def hidden_edges(edges,retained):
 """Consumer norm.saved still contains the exact incoming hidden after math."""
 import numpy as np
 results=[]
 for edge in edges:
  a=retained['hidden',tuple(edge['source'])];b=retained['norm_saved',tuple(edge['destination'])]
  if a.dtype!=np.dtype('<u2') or b.dtype!=a.dtype or a.size!=5120 or b.size!=5120:raise ValueError('Full boundary hidden native BF16')
  if not a.flags.c_contiguous or not b.flags.c_contiguous:raise ValueError('Contiguous retained vectors')
  if not np.array_equal(a.reshape(-1),b.reshape(-1)):raise ValueError('Source final hidden differs from consumer actual saved input')
  results.append(dict(source=edge['source'],destination=edge['destination'],BF16_words=5120,native_sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest()))
 return results

def commit(root,host_plan,lowered,captures,fence,artifact_sha256,prepared_pins):
 root=Path(root);expected=host_plan['copies']['checkpoint_copies'];path=root/'checkpoint-pos0.json'
 if path.exists() or len(captures)!=len(expected):raise ValueError('Fresh complete checkpoint only')
 required=dict(PEs=138040,serial=1,generation=1,next_position=1,transport_snapshots=2,stable=True,all_commands_complete=True,all_boundary_transfers_complete=True)
 if any(fence.get(k)!=v for k,v in required.items()) or len(fence.get('hidden_edges',[]))!=3:raise ValueError('Verified complete causal global fence required')
 for got,edge in zip(fence['hidden_edges'],lowered['hidden_edges']):
  if got['source']!=edge['source'] or got['destination']!=edge['destination'] or got['BF16_words']!=5120 or len(got['native_sha256'])!=64:raise ValueError('Exact three retained hidden edge comparisons')
 if len(artifact_sha256)!=64 or set(prepared_pins)!={'0','1','2','3'}:raise ValueError('Exact artifact and all four original parameter preparations')
 records=[];total=0;seen=set()
 for item,(got,receipt) in zip(expected,captures):
  if identity(item)!=identity(got) or identity(item) in seen:raise ValueError('Exact unique ordered checkpoint geometry')
  seen.add(identity(item));value=_payload(root,item,receipt,False)
  if item['symbol']=='checkpoint_control':control(value,lowered['regions'],lowered['request_id'],lowered['stage_id'])
  payload=root/receipt['path'];payload.chmod(0o444);total+=receipt['bytes']
  records.append(dict(copy=item,receipt=receipt))
 if total!=host_plan['checkpoint_native_bytes'] or total!=14378752:raise ValueError('Every hidden/conv/DeltaNet/KV/control byte')
 value=dict(schema=1,status='durable_device_checkpoint_after_global_fence',model=lowered['model'],revision=lowered['revision'],precision=lowered['precision'],lowered_input_sha256=lowered['input_sha256'],artifact_sha256=artifact_sha256,prepared_pins=prepared_pins,request_id=lowered['request_id'],stage_id=lowered['stage_id'],generation=1,next_position=1,records=records,total_native_bytes=total,fence=fence,requires_independent_baseline_runtime_release=True,actual_restore_qualified=False)
 raw=_json_bytes(value)
 if len(raw)>128<<10:raise ValueError('Bounded checkpoint metadata')
 temporary=root/'checkpoint-pos0.json.pending'
 with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 temporary.chmod(0o444);os.link(temporary,path);temporary.unlink()
 for directory in (root/'raw',root):
  fd=os.open(directory,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 return dict(path=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),files=len(records),native_bytes=total)

class Restore:
 def __init__(self,root,pin,host_plan,lowered,artifact_sha256,prepared_pins):
  self.root=Path(root);p=self.root/'checkpoint-pos0.json';before=_stamp(p)
  if p.is_symlink() or before[5]&0o222 or before[2]>128<<10:raise ValueError('Immutable bounded checkpoint manifest')
  raw=p.read_bytes()
  if dict(path=p.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),files=39,native_bytes=14378752)!=pin or _stamp(p)!=before:raise ValueError('Exact durable checkpoint pin')
  d=json.loads(raw)
  for key,v in dict(model=lowered['model'],revision=lowered['revision'],precision=lowered['precision'],lowered_input_sha256=lowered['input_sha256'],artifact_sha256=artifact_sha256,prepared_pins=prepared_pins,generation=1,next_position=1).items():
   if d[key]!=v:raise ValueError('Restore identity mismatch: '+key)
  expected=host_plan['copies']['checkpoint_copies']
  if len(d['records'])!=len(expected) or d['total_native_bytes']!=host_plan['checkpoint_native_bytes']:raise ValueError('Full restore inventory')
  self.records=[];self.stamps={p:before};self.lowered=lowered
  for item,record in zip(expected,d['records']):
   if identity(item)!=identity(record['copy']):raise ValueError('Restore geometry changed')
   value=_payload(self.root,item,record['receipt'],True)
   if item['symbol']=='checkpoint_control':control(value,lowered['regions'],lowered['request_id'],lowered['stage_id'])
   self.stamps[self.root/record['receipt']['path']]=_stamp(self.root/record['receipt']['path']);self.records.append((item,record['receipt']))
  self.finished=set()
 def transfer(self,index):
  if index in self.finished or not 0<=index<len(self.records):raise ValueError('One restore view per exact bank')
  item,receipt=self.records[index];value=_payload(self.root,item,receipt,True);self.finished.add(index)
  return item,value
 def verify_unchanged(self):
  if self.finished!=set(range(len(self.records))):raise ValueError('All restore bank views consumed')
  for path,before in self.stamps.items():
   if _stamp(path)!=before:raise ValueError('Checkpoint changed while restoring')
  return dict(files=len(self.records),native_bytes=14378752,all_checkpoint_views_consumed=True,physical_restore_qualified=False)
