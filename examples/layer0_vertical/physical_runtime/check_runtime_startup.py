"""Finite fake-clock startup fault checks; no SDK or model payload."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,tempfile
from hw00.store import atomic_json
from runtime_startup import Parent,Publisher
from runtime_monitor import Monitor
from startup_trace import StartupTrace

PROFILE=json.loads((Path(__file__).resolve().parent/'runtime-profile.json').read_bytes())['runtime']
cases=[]


def reject(label,fn,error):
    try:fn()
    except error:cases.append(label)
    else:raise AssertionError(label)


def record(root,upload=None,start=None,finish=None):
    atomic_json(root/'startup-segments.json',dict(schema=1,upload_started=upload,
        device_start_started=start,device_start_finished=finish))


with tempfile.TemporaryDirectory() as name:
    root=Path(name);clock=[0.];parent=Parent(root,PROFILE,0,clock=lambda:clock[0])
    clock[0]=240.;parent.check(None)
    clock[0]=240.001
    reject('Local preparation plus management still ends at240s without a start call',lambda:parent.check(None),TimeoutError)
    record(root,100.);parent=Parent(root,PROFILE,0,clock=lambda:clock[0])
    reject('Upload activity cannot renew the pre-start240s window',lambda:parent.check('startup'),TimeoutError)
    record(root,170.13,171.14);clock[0]=300.;parent=Parent(root,PROFILE,0,clock=lambda:clock[0])
    parent.check('startup');cases.append('Actual171.14s start retains its own180s window beyond old240s total')
    clock[0]=351.141
    reject('Blocked device RPC expires180s after its first observed call',lambda:parent.check('startup'),TimeoutError)
    clock[0]=200.;record(root,170.13,180.)
    reject('Changed first device-start anchor cannot renew the deadline',lambda:parent.check('startup'),ValueError)

with tempfile.TemporaryDirectory() as name:
    root=Path(name);clock=[400.]
    record(root,170.,171.,352.)
    reject('A completed over-budget RPC is rejected even when intermediate polls were missed',
        lambda:Parent(root,PROFILE,0,clock=lambda:clock[0]).check('capture'),TimeoutError)
    record(root,200.,241.,300.)
    reject('Late pre-start transition is rejected even after device return',
        lambda:Parent(root,PROFILE,0,clock=lambda:clock[0]).check('capture'),TimeoutError)
    record(root,170.,171.)
    clock[0]=200.
    reject('Capture cannot precede the observed device-start return',
        lambda:Parent(root,PROFILE,0,clock=lambda:clock[0]).check('capture'),ValueError)
    record(root,None,171.)
    reject('Device start cannot appear before upload anchor',
        lambda:Parent(root,PROFILE,0,clock=lambda:clock[0]).check('startup'),ValueError)
    # Deliberately bypass the writer's allow_nan=False to test parent rejection.
    (root/'startup-segments.json').write_text(json.dumps(dict(schema=1,upload_started=float('nan'),
        device_start_started=None,device_start_finished=None)))
    reject('Nonfinite startup timestamp is rejected',
        lambda:Parent(root,PROFILE,0,clock=lambda:clock[0]).check('startup'),ValueError)
    clock[0]=400.;record(root,170.,171.,300.);parent=Parent(root,PROFILE,0,clock=lambda:clock[0]);parent.check('capture')
    (root/'startup-segments.json').unlink()
    reject('Previously observed segment record cannot disappear',lambda:parent.check('capture'),ValueError)

with tempfile.TemporaryDirectory() as name:
    root=Path(name);pub=Publisher(root)
    reject('Publisher rejects start before artifact upload',lambda:pub.observe('SdkRuntime.start','call',171.),ValueError)
    pub.observe('SdkClient.upload_artifact','call',170.)
    reject('Publisher rejects repeated artifact upload',lambda:pub.observe('SdkClient.upload_artifact','call',171.),ValueError)
    pub.observe('SdkRuntime.start','call',171.)
    reject('Publisher rejects repeated device start',lambda:pub.observe('SdkRuntime.start','call',172.),ValueError)
    pub.observe('SdkRuntime.start','return',300.)
    assert json.loads((root/'startup-segments.json').read_bytes())['device_start_finished']==300.
    cases.append('One upload/start/return records three ordered immutable anchors')

with tempfile.TemporaryDirectory() as name:
    root=Path(name);clock=[420.];record(root,239.,240.,420.)
    atomic_json(root/'runtime-phase.json',dict(phase='startup',started=0.,deadline=420.))
    monitor=Monitor(root,PROFILE,0.,clock=lambda:clock[0]);monitor.check(420.)
    clock[0]=420.001
    reject('Integrated parent retains overall420s startup ceiling after device return',lambda:monitor.check(clock[0]),TimeoutError)
    clock[0]=1080.001
    reject('Integrated parent always retains1080s physical-client ceiling',lambda:monitor.check(clock[0]),TimeoutError)

with tempfile.TemporaryDirectory() as name:
    root=Path(name)
    def upload():pass
    def start():pass
    class Store:
        def __init__(self):self.rows=[]
        def event(self,kind,**fields):self.rows.append(dict(kind=kind,**fields))
    store=Store();trace=StartupTrace(root,store,{upload.__code__:'SdkClient.upload_artifact',start.__code__:'SdkRuntime.start'},PROFILE)
    trace.started=0.
    for function,event,now in [(upload,'call',170.),(upload,'return',171.),(start,'call',172.),(start,'return',300.)]:
        with patch('startup_trace.time.monotonic',return_value=now):trace.callback(SimpleNamespace(f_code=function.__code__),event,None)
    assert trace.events==4 and not trace.stack and len(store.rows)==4
    assert json.loads((root/'startup-segments.json').read_bytes())==dict(schema=1,upload_started=170.,device_start_started=172.,device_start_finished=300.)
    cases.append('Existing profiling callback publishes segments while preserving ordinary trace events')

report=dict(status='startup_source_fault_checks_passed_no_SDK',count=len(cases),cases=cases,
    profile={k:PROFILE[k] for k in ('startup_pre_start_seconds','startup_device_seconds','startup_seconds','runtime_seconds')},
    SDK_invocations=0,hardware_invocations=0,model_payload_bytes=0)
(Path(__file__).resolve().parent/'STARTUP-SOURCE-CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
