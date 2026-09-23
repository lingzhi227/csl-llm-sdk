"""Observe finite pre-start and device-start windows without changing SDK calls."""
import json,math,time
from pathlib import Path
from hw00.store import atomic_json

FIELDS=('upload_started','device_start_started','device_start_finished')


def stamp(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Finite startup segment timestamp')
    return value


class Publisher:
    def __init__(self,root):
        self.path=Path(root)/'startup-segments.json'
        self.value=dict(schema=1,**{name:None for name in FIELDS})

    def observe(self,name,event,now):
        target={('SdkClient.upload_artifact','call'):'upload_started',
                ('SdkRuntime.start','call'):'device_start_started',
                ('SdkRuntime.start','return'):'device_start_finished'}.get((name,event))
        if target is None:return
        index=FIELDS.index(target);stamp(now)
        if self.value[target] is not None:raise ValueError('Startup segment cannot be repeated')
        if index and self.value[FIELDS[index-1]] is None:raise ValueError('Startup segment order')
        if index and now<self.value[FIELDS[index-1]]:raise ValueError('Startup segment time order')
        self.value[target]=now;atomic_json(self.path,self.value)


class Parent:
    def __init__(self,root,profile,started,clock=None):
        self.path=Path(root)/'startup-segments.json';self.started=stamp(started)
        self.clock=time.monotonic if clock is None else clock
        self.limits=tuple(profile[name] for name in ('startup_pre_start_seconds','startup_device_seconds'))
        if self.limits!=(240,600) or profile['startup_seconds']!=840:
            raise ValueError('Exact independently reviewed startup segments')
        self.seen={};self.completed=False

    def check(self,phase):
        # Re-read even after capture begins to reject changed or removed anchors.
        value={name:None for name in FIELDS}
        if self.path.exists():
            if self.path.is_symlink() or self.path.stat().st_size>4096:
                raise ValueError('Small regular startup segment record')
            row=json.loads(self.path.read_bytes())
            if not isinstance(row,dict) or set(row)!={'schema',*FIELDS} or type(row['schema']) is not int or row['schema']!=1:
                raise ValueError('Exact startup segment record schema')
            value={name:row[name] for name in FIELDS}
        now=self.clock();prior=self.started;missing=False
        for name in FIELDS:
            current=value[name]
            if current is None:
                if name in self.seen:raise ValueError('Startup anchor disappeared')
                missing=True;continue
            current=stamp(current)
            if missing or current<prior or current>now:raise ValueError('Ordered startup timestamps')
            if name in self.seen and self.seen[name]!=current:
                raise ValueError('Startup deadline cannot be renewed')
            self.seen[name]=current;prior=current
        upload,start,finish=(value[name] for name in FIELDS)
        if (now if start is None else start)-self.started>self.limits[0]:
            raise TimeoutError('Preparation, management and artifact pre-start deadline')
        if start is not None and (now if finish is None else finish)-start>self.limits[1]:
            raise TimeoutError('Device start startup deadline')
        if finish is not None and finish-self.started>840:
            raise TimeoutError('Completed startup exceeded overall deadline')
        if phase in ('capture','stop','stopped') and finish is None:
            raise ValueError('Capture or stop requires completed observed SDK startup')
        self.completed=finish is not None
