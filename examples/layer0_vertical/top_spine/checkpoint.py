"""Actual complete Layer0 payload conversion; protocol restore is separate.

Callers supply native arrays from the two exact declared export ROIs. No
nominal/reference state is used. Private request/position restoration must be
implemented and qualified before these payloads can resume a stage context.
"""
def canonical(copies,conv_roi,recurrent_roi):
 import numpy as np
 conv,rec=copies
 assert conv['symbol']=='conv_history' and rec['symbol']=='recurrent_matrix'
 assert conv_roi.dtype==np.dtype('<u2') and list(conv_roi.shape)==conv['host_shape']
 assert recurrent_roi.dtype==np.dtype('<f4') and list(recurrent_roi.shape)==rec['host_shape']
 return dict(conv_history=conv_roi.reshape(10240,4).copy(),
  recurrent_matrix=recurrent_roi.reshape(48,4,128,32).transpose(0,2,1,3).reshape(48,128,128).copy())
def physical(copies,checkpoint):
 import numpy as np
 conv=checkpoint['conv_history'];rec=checkpoint['recurrent_matrix']
 assert conv.dtype==np.dtype('<u2') and conv.shape==(10240,4)
 assert rec.dtype==np.dtype('<f4') and rec.shape==(48,128,128)
 return [conv.reshape(copies[0]['host_shape']).copy(),
  rec.reshape(48,128,4,32).transpose(0,2,1,3).reshape(copies[1]['host_shape']).copy()]
