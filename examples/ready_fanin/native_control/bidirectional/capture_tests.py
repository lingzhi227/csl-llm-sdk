"""Focused physical adapter and full original saved-array checks, SDK-free."""
from pathlib import Path
from copy import deepcopy
import ast,hashlib,json,os,sys,types
import numpy as np
from checker_tests import fixture
from check_lifecycle import ROUTING
from capture import capture
from validate_capture import captured_result
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'host-mock-results';assert not OUT.exists();OUT.mkdir()
assert sorted(os.sched_getaffinity(0))==[0]
CURRENT={}
class Runner:
    def __init__(self):self.phase=0;self.snapshots=0;self.stops=0;self.copies=0
    def get_id(self,name):return name
    def load(self):raise AssertionError("Physical adapter must not invoke simulator load")
    def run(self):raise AssertionError("Physical adapter must not invoke simulator run")
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
            rows=[[1,i]+[0]*62 for i in range(3)] if self.phase==1 else deepcopy(data['state'])
            if self.phase==2 and self.snapshots<CURRENT['settle_poll']:rows[0][48]=1
        elif symbol=='routing':rows=[ROUTING+[4] for _ in range(3)]
        else:rows=data[{'observed':'observed','tx_proof':'tx_proof','received_buffer':'rx_banks',
                         'transmit_frame':'live_frames','current_source':'live_sources','row_archive':'row_archive'}[symbol]]
        value=np.asarray(rows,dtype=np.uint32).reshape(-1)
        assert x==0 and y==0 and width==3 and height==1 and value.size==destination.size==3*count
        destination[:]=value
    def stop(self):
        self.stops+=1;assert self.stops==1
        if CURRENT['stop_error']:raise RuntimeError('Intended mock stop failure')

class FakeSDK:
    def __init__(self,*args,**kwargs):
        assert kwargs==dict(simulator=False,disable_version_check=True,resource_cpu=1000,resource_mem=4<<30)
        self.runner=Runner();self.entries=0;self.exits=0
    def __enter__(self):self.entries+=1;assert self.entries==1;return self
    def __exit__(self,*args):
        self.exits+=1;assert self.exits==1 and args==(None,None,None);self.runner.stop()
    def __getattr__(self,name):return getattr(self.runner,name)

source=(ROOT/'run_hw.py').read_text();tree=ast.parse(source)
node=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='Runtime')
module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
namespace=dict(SdkRuntime=FakeSDK,artifact=Path('never-opened-mock-artifact'),PROFILE=dict(worker_cpu_millicores=1000,worker_memory_bytes=4<<30),mark=lambda kind:None)
exec(compile(module,'actual-physical-context-class','exec'),namespace)
Runtime=namespace['Runtime']
delta=json.loads((ROOT/'PHYSICAL-HOST-DELTA.json').read_bytes())
raw=(ROOT/'simulator-capture-original.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==delta['parent_capture_sha256']
expected=raw.decode()
for edit in delta['replacements']:
    assert expected.count(edit['old'])==1;expected=expected.replace(edit['old'],edit['new'])
assert expected==(ROOT/'capture.py').read_text()
assert hashlib.sha256(expected.encode()).hexdigest()==delta['candidate_capture_sha256']
parent=(ROOT/'simulator-entry-original.py').read_text();parent_ast=ast.parse(parent)
node=next(n for n in parent_ast.body if isinstance(n,ast.FunctionDef) and n.name=='captured_result')
original=ast.get_source_segment(parent,node).replace("result['hardware'] is False","result['hardware'] is True")
actual=next(n for n in ast.parse((ROOT/'validate_capture.py').read_text()).body if isinstance(n,ast.FunctionDef))
assert ast.dump(ast.parse(original).body[0],include_attributes=False)==ast.dump(actual,include_attributes=False)
DT=types.SimpleNamespace(MEMCPY_32BIT=1);ORDER=types.SimpleNamespace(ROW_MAJOR=1);results={}
for name,settle_poll,copy_error,stop_error in [('full_budget_success',8,False,False),('first_poll_success',1,False,False),('copy_failure',8,True,False),('stop_failure',8,False,True)]:
    CURRENT.update(data=fixture(3),settle_poll=settle_poll,copy_error=copy_error,stop_error=stop_error)
    root=OUT/name;root.mkdir();runner=Runtime();error=None
    assert runner.runtime.entries==1 and runner.runtime.exits==0 and not runner.closed
    try:capture(root,lambda:runner,DT,ORDER)
    except RuntimeError as exc:error=str(exc)
    result=json.loads((root/'result.json').read_bytes())
    assert bool(error)==(copy_error or stop_error)
    assert runner.runtime.exits==runner.runtime.runner.stops==1 and runner.stop_attempted
    assert runner.closed==result['normal_stop']==(not stop_error)
    assert result['hardware'] is True and result['H2D']==0
    assert result['status']==('failed' if error else 'passed')
    if not copy_error:
        assert result['copies']==settle_poll+9 and result['host_bytes']==11976+768*settle_poll and result['launches']==settle_poll+3
    else:assert result['copies']==3 and result['host_bytes']==1632 and result['launches']==4
    if not error:assert captured_result(root)==result
    if stop_error:
        try:runner.stop();raise AssertionError('Failed context exit must not retry')
        except RuntimeError as exc:assert 'already attempted' in str(exc)
    else:runner.stop()
    assert runner.runtime.exits==1
    results[name]=dict(status=result['status'],normal_stop=result['normal_stop'],copies=result['copies'],host_bytes=result['host_bytes'],launches=result['launches'],context_entries=runner.runtime.entries,context_exits=runner.runtime.exits)
assert not any(name.startswith('cerebras') for name in sys.modules)
print(json.dumps(dict(passed=True,cases=results,original_saved_array_validator=True,actual_context_class=True,exact_host_delta=True,SDK_imports=0,SDK_invocations=0,hardware_invocations=0)))
