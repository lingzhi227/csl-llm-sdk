"""Bounded, startup-only observation of the unchanged installed SDK calls.

Uses Python profiling callbacks without replacing SDK methods or controlling
their arguments, returns, RPC retries, or order. Never enabled during capture.
"""
import faulthandler,hashlib,resource,sys,time
from pathlib import Path
from hw00.store import atomic_json
from runtime_startup import Publisher

SDK_SOURCE_SHA='98ec511c207fc3a289d59bb7f9b14859a4fbd0a29a9e101710a6b2cffcb45eaa'
TARGETS={
    'SdkContext':('__enter__','_init_mgmt_client','_init_appliance_stub'),
    'SdkClient':('__enter__','upload_artifact','download_artifact','_upload_files'),
    'SdkRuntime':('__init__','__enter__','start'),
}


def memory_snapshot():
    fields=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return dict(**{key:int(fields.get(key,'0').split()[0])*1024 for key in
                ('VmSize','VmPeak','VmRSS','VmHWM','VmData','VmStk','VmSwap')},
                Threads=int(fields['Threads']),RLIMIT_AS=list(resource.getrlimit(resource.RLIMIT_AS)))


def sdk_targets(module):
    path=Path(module.__file__)
    if path.stat().st_size!=58687 or hashlib.sha256(path.read_bytes()).hexdigest()!=SDK_SOURCE_SHA:
        raise ValueError('Exact observed installed SDK source identity')
    targets={}
    for cls,names in TARGETS.items():
        for name in names:
            function=getattr(getattr(module,cls),name)
            code=function.__code__
            if code in targets:raise ValueError('Distinct exact SDK startup target')
            targets[code]=cls+'.'+name
    return targets


class StartupTrace:
    def __init__(self,root,store,targets,profile):
        self.root=Path(root);self.store=store;self.targets=targets;self.profile=profile
        self.events=0;self.stack=[];self.file=None;self.started=None
        self.segments=Publisher(root)

    def callback(self,frame,event,arg):
        if event not in ('call','return') or frame.f_code not in self.targets:return
        if self.events>=self.profile['startup_trace_events']:raise ValueError('Finite startup trace event cap')
        now=time.monotonic();name=self.targets[frame.f_code];self.events+=1
        self.segments.observe(name,event,now)
        if event=='call':self.stack.append((frame.f_code,name,now))
        else:
            if not self.stack or self.stack[-1][0]!=frame.f_code:raise ValueError('Startup trace nesting')
            self.stack.pop()
        # No arguments, locals, credentials, or payload bytes are serialized.
        active=[dict(name=label,started=start) for _,label,start in self.stack]
        self.store.event('startup_step_'+event,name=name,monotonic=now,
            since_trace_start=now-self.started,active=active,memory=memory_snapshot())
        atomic_json(self.root/'startup-progress.json',dict(events=self.events,active=active,
            last_event=event,last_step=name,monotonic=now))

    def __enter__(self):
        if sys.getprofile() is not None:raise ValueError('No pre-existing Python profiler')
        self.started=time.monotonic();self.file=(self.root/'startup-stack.log').open('xb',buffering=0)
        self.store.event('startup_trace_enter',monotonic=self.started,
            single_stack_dump_after_seconds=self.profile['startup_stack_after_seconds'],memory=memory_snapshot())
        faulthandler.dump_traceback_later(self.profile['startup_stack_after_seconds'],repeat=False,file=self.file,exit=False)
        sys.setprofile(self.callback)
        return self

    def __exit__(self,kind,value,tb):
        sys.setprofile(None);faulthandler.cancel_dump_traceback_later()
        self.file.close()
        self.store.event('startup_trace_closed',events=self.events,exception=kind is not None,
            elapsed_seconds=time.monotonic()-self.started,memory=memory_snapshot())
        return False
