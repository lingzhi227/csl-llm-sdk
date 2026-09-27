"""Resident actor correctness/lifetime qualification, no throughput extrapolation."""
import hashlib,json
from pathlib import Path
import numpy as np
from backend import runtime
meta=json.loads(Path('fixture.json').read_text());assert hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest()==meta['fixture_sha256']
with np.load('fixture.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
captured={}
with runtime(False) as (runner,dtype,order):
 ids={n:runner.get_id(n) for n in ['bank','values','packet','audit']}
 opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False,data_type=dtype.MEMCPY_32BIT)
 def upload(name,xy,value):
  a=np.ascontiguousarray(value).view(np.uint32);runner.memcpy_h2d(ids[name],a,xy[0],xy[1],1,1,len(a),**opts)
 def read(name,xy,n):
  a=np.zeros(n,np.uint32);runner.memcpy_d2h(a,ids[name],xy[0],xy[1],1,1,n,**opts);return a
 for xy in meta['helpers']:upload('bank',xy,data['bank'])
 for case in range(3):
  for xy,value in zip([(0,1),(1,2),(2,0)],data['payload'][case]):upload('values',xy,value)
  upload('packet',(0,0),data['packet'][case]);runner.launch('arm',nonblock=False)
  for y in range(3):
   for x in range(3):assert read('audit',(x,y),1)[0]==1
  runner.launch('start',nonblock=False)
  for name,xy,expected in [('collector',(1,1),data['partial'][case]),('header',(1,0),data['expected'][case]),('sink',(0,0),data['expected'][case])]:
   actual=read('values',xy,128);captured[f'{name}_{case}']=actual
   np.savez('actual.npz',**captured)
   np.testing.assert_array_equal(actual,expected.view(np.uint32))
  np.testing.assert_array_equal(read('packet',(1,2),65),data['packet'][case])
  np.testing.assert_array_equal(read('packet',(0,0),65),data['packet'][case])
  for y in range(3):
   for x in range(3):
    expected=np.zeros(12,np.uint32);expected[0]=expected[11]=1
    if (x,y)==(1,1):expected[1:5]=16
    if (x,y)==(1,0):expected[1:5]=[16,16,17,16];expected[5:7]=1
    if (x,y) in [(0,1),(2,0)]:expected[3]=1
    if (x,y)==(1,2):expected[3]=expected[5]=1
    if (x,y)==(0,0):expected[1]=expected[3]=1
    actual=read('audit',(x,y),12);captured[f'audit_{case}_{x}_{y}']=actual
    np.testing.assert_array_equal(actual,expected)
 for slot,half in [(0,0),(0,1),(67,0),(67,1)]:
  runner.launch('embedding_lookup',np.uint16(slot),np.uint16(half),nonblock=False)
  for xy in meta['helpers']:
   np.testing.assert_array_equal(read('values',xy,64),data['bank'][slot*128+half*64:slot*128+(half+1)*64])
 for xy in meta['helpers']:np.testing.assert_array_equal(read('bank',xy,8814),data['bank'])
 np.savez('actual.npz',**captured)
result=dict(passed=True,normal_stop=True,physical=False,full_model=False,fixture_sha256=meta['fixture_sha256'],
            epochs=3,partial_and_final_values_checked=1152,all_counters_exact=True,all_original_banks_retained=True,
            embedding_lookup_words_checked=512,ingress_relay_words_checked=390,scope=meta['scope'],measured_speedup=None)
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
