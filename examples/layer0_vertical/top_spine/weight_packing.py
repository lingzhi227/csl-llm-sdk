"""Repack accepted original words into bounded physical ROW_MAJOR ROIs.

Source files retain logical [1,ordinal,6144] u32 order. This does not transpose
an original matrix or alter a single word/halfword inside a128x96 tile.
"""
import hashlib

def pack_roi(item,load_accepted_stripe):
 import numpy as np
 assert item['count']==6144 and item['host_bytes']<=16<<20 and item['order']=='ROW_MAJOR'
 out=np.zeros(tuple(item['host_shape']),dtype='<u4');covered=np.zeros(out.shape[:2],dtype=np.bool_)
 for segment in item['segments']:
  source=load_accepted_stripe(segment['logical_stripe']);begin=segment['ordinal_begin'];n=segment['tile_count']
  x,y=segment['destination_x'],segment['destination_y']
  assert source.dtype==np.dtype('<u4') and source.ndim==3 and source.shape[0]==1 and source.shape[2]==6144
  assert begin+n<=source.shape[1] and y+n<=out.shape[0] and x<out.shape[1] and not np.any(covered[y:y+n,x])
  out[y:y+n,x,:]=source[0,begin:begin+n,:];covered[y:y+n,x]=True
 assert np.all(covered)
 return out

def validate_roi(item,packed,load_accepted_stripe):
 """Independent per-PE word comparison against separately reloaded pinned stripes."""
 import numpy as np
 assert packed.dtype==np.dtype('<u4') and list(packed.shape)==item['host_shape']
 checks=0;original=[]
 for segment in item['segments']:
  source=load_accepted_stripe(segment['logical_stripe']);begin=segment['ordinal_begin'];n=segment['tile_count']
  for ordinal in range(begin,begin+n):
   actual=packed[segment['destination_y']+ordinal-begin,segment['destination_x'],:]
   expected=source[0,ordinal,:]
   assert actual.tobytes()==expected.tobytes();checks+=1
  original.append(dict(logical_stripe=segment['logical_stripe'],ordinal_begin=begin,tile_count=n,
   native_sha256=hashlib.sha256(source[0,begin:begin+n,:].tobytes()).hexdigest()))
 assert checks==item['width']*item['height']
 return dict(passed=True,tiles=checks,unchanged_u32_words=checks*6144,source_segments=original)
