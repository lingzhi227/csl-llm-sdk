"""Concrete 8-PE qualification copies; one device stage invocation per input."""
from .resident_spatial import TILES,arrays

def sequence():
    operations=[]
    def launch(name,g):operations.append(dict(kind='launch',name=name,generation=g))
    def copy(name,rank,g,direction='d2h',whole=False):
        n,kind=arrays(TILES[rank])[name];tile=TILES[rank]
        w,h=(4,2) if whole else (1,1)
        if whole:
            if len({arrays(t)[name] for t in TILES})!=1:raise ValueError('Uniform multi-PE export extent')
            x=y=0
        else:x,y=tile.x,tile.y
        count=n*w*h;bits=16 if kind=='u16' else 32
        operations.append(dict(kind='copy',name=name,rank=None if whole else rank,generation=g,
            direction=direction,x=x,y=y,width=w,height=h,count_per_PE=n,count=count,
            bits=bits,dtype='float32' if kind=='f32' else 'uint32',host_bytes=4*count,native_bytes=(bits//8)*count))
    launch('initialize',0)
    for t in TILES:
        if t.matrix:copy('weights',t.rank,0,'h2d')
    copy('state',0,0,whole=True)
    for g in range(1,4):
        copy('request',4,g,'h2d');launch('stage',g)
        # Blocking controller output D2H is the first result fence. Each PE also
        # holds its host command stream until its own local release completes.
        copy('output',4,g);copy('state',0,g,whole=True);copy('timing',0,g,whole=True)
        for name in ('input','partial','reduced'):
            for t in TILES:
                if t.matrix:copy(name,t.rank,g)
        for t in TILES:
            if t.root:copy('rounded',t.rank,g)
        for name in ('gate','up','silu','product','exponential','sigmoid','activation_fp32','product_fp32'):copy(name,5,g)
        copy('broadcast',4,g);copy('broadcast',5,g);copy('request',4,g)
    for t in TILES:
        if t.matrix:copy('weights',t.rank,3)
    for i,op in enumerate(operations):op['sequence']=i
    return operations

def accounting():
    ops=sequence();copies=[o for o in ops if o['kind']=='copy']
    return dict(operations=len(ops),copies=len(copies),launches=len(ops)-len(copies),
        H2D=sum(o['direction']=='h2d' for o in copies),D2H=sum(o['direction']=='d2h' for o in copies),
        host_bytes=sum(o['host_bytes'] for o in copies),native_bytes=sum(o['native_bytes'] for o in copies),
        H2D_host_bytes=sum(o['host_bytes'] for o in copies if o['direction']=='h2d'),
        D2H_host_bytes=sum(o['host_bytes'] for o in copies if o['direction']=='d2h'),
        raw_observation_bytes=sum(o['host_bytes'] for o in copies if o['direction']=='d2h'),
        arrays_per_generation=35,initial_resident_weight_uploads=6,steady_token_weight_uploads=0,
        final_resident_weight_readbacks=6,host_neural_intermediate_roundtrips=0,
        diagnostic_arrays_do_not_drive_device_compute=True)

def expected_state(rank,generation):
    result=[0]*32;result[31]=1
    if generation==0:return result
    t=TILES[rank];result[1]=generation;result[9]=generation
    if rank==4:
        result[2]=result[3]=result[4]=0xef;result[5]=1;result[11]=3*generation
    elif rank==5:result[5]=1;result[8]=generation;result[11]=6*generation
    elif t.matrix:
        result[6]=result[7]=generation
        if t.root:result[10]=3*generation
    return result
