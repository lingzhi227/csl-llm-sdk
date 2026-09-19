"""Bounded physical adapter tests; no SDK imports or captures on hardware."""
from pathlib import Path
from copy import deepcopy
import hashlib,json,os,sys,types
import numpy as np
from check_multisource import synthetic,ROUTING
from capture import capture
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'host-mock-results';assert not OUT.exists();OUT.mkdir()
CURRENT={};instances=[]
class Runner:
    def __init__(self,*args):self.phase=0;self.snapshots=0;self.stops=0;self.copies=0;self.closed=False;instances.append(self)
    def get_id(self,name):return name
    def load(self):pass
    def run(self):pass
    def launch(self,name,**kwargs):
        if name=='initialize':self.phase=1
        elif name=='compute':self.phase=2
        elif name=='snapshot':self.snapshots+=1
        else:raise AssertionError(name)
    def memcpy_d2h(self,destination,symbol,x,y,width,height,count,**kwargs):
        self.copies+=1
        if CURRENT['copy_error'] and self.copies==4:raise RuntimeError('Intended mock copy failure')
        data=CURRENT['data']
        if symbol=='state':
            rows=[[1,i]+[0]*46 for i in range(3)] if self.phase==1 else deepcopy(data['state'])
            if self.phase==2 and self.snapshots<8:rows[0][23]=3
        elif symbol=='routing':rows=[ROUTING+[4] for _ in range(3)]
        elif symbol=='observed':rows=data['observed']
        elif symbol=='tx_proof':rows=[[0]*128,*data['tx_proof']]
        elif symbol=='received_buffer':rows=data['rx_banks']
        else:raise AssertionError(symbol)
        value=np.asarray(rows[x:x+width],dtype=np.uint32).reshape(-1)
        assert value.size==destination.size==width*count;destination[:]=value
    def stop(self):
        self.stops+=1;assert self.stops==1
        if CURRENT['stop_error']:raise RuntimeError('Intended mock stop failure')
        self.closed=True

raw=(ROOT/'simulator-driver-original.py').read_bytes();delta=json.loads((ROOT/'PHYSICAL-HOST-DELTA.json').read_bytes())
assert hashlib.sha256(raw).hexdigest()==delta['parent_sha256']
expected=raw.decode()
for edit in delta['replacements']:
    assert expected.count(edit['old'])==1;expected=expected.replace(edit['old'],edit['new'])
assert expected==(ROOT/'capture.py').read_text()
assert hashlib.sha256(expected.encode()).hexdigest()==delta['candidate_sha256']
DT=types.SimpleNamespace(MEMCPY_32BIT=1);ORDER=types.SimpleNamespace(ROW_MAJOR=1)
results={}
for name,copy_error,stop_error in [('full_budget_success',False,False),('copy_failure',True,False),('stop_failure',False,True)]:
    CURRENT.update(data=synthetic(),copy_error=copy_error,stop_error=stop_error);root=OUT/name;root.mkdir();runner=Runner();error=None
    try:capture(root,runner,DT,ORDER)
    except RuntimeError as exc:error=str(exc)
    result=json.loads((root/'result.json').read_bytes())
    assert bool(error)==(copy_error or stop_error) and runner.stops==1
    assert result['normal_stop']==(not stop_error) and result['hardware'] is True and result['H2D']==0
    assert result['status']==('failed' if error else 'passed')
    if not copy_error:assert result['copies']==14 and result['host_bytes']==7748 and result['launches']==11
    else:assert result['copies']==3 and result['host_bytes']==1248 and result['launches']==4
    if not error:assert all(row['passed'] for row in result['checks'].values())
    progress=json.loads((root/'capture-progress.json').read_bytes());assert progress['captures']==result['captures']
    for receipt in result['captures']:
        q=root/receipt['path'];raw=q.read_bytes()
        assert len(raw)==receipt['bytes']<=4096 and hashlib.sha256(raw).hexdigest()==receipt['sha256']
        with np.load(q,allow_pickle=False) as archive:
            assert archive.files==['words'] and archive['words'].dtype==np.uint32
            assert archive['words'].size==receipt['shape'][0]*receipt['shape'][1]
    window=json.loads((root/'monitor-deadline.json').read_bytes());assert window['active']==stop_error
    results[name]=dict(status=result['status'],normal_stop=result['normal_stop'],copies=result['copies'],host_bytes=result['host_bytes'],launches=result['launches'],stop_calls=runner.stops)
assert not any(name.startswith('cerebras') for name in sys.modules)
print(json.dumps(dict(passed=True,cases=results,exact_host_delta=True,SDK_imports=0,SDK_invocations=0,hardware_invocations=0)))
