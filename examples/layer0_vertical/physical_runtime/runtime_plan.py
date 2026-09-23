"""Metadata-only exact copy geometry and finite three-position capture budget."""
from collections import defaultdict
from host_roles import roles

PARAMETERS={'weights','norm_gains','conv_weights','head_gains','head_parameters'}
CONTROL={'transport_config','route','descriptor','state','transport_status','observer'}

def rectangles(points):
    """Combine equal-extent adjacent banks into rectangles without role holes."""
    rows=defaultdict(list);runs=defaultdict(list)
    for (x,y),count in points.items():rows[y].append((x,count))
    for y,values in sorted(rows.items()):
        start=last=count=None
        for x,n in sorted(values):
            if start is not None and (x!=last+1 or n!=count):
                runs[start,last-start+1,count].append(y);start=None
            if start is None:start=x;count=n
            last=x
        if start is not None:runs[start,last-start+1,count].append(y)
    for (x,width,count),ys in sorted(runs.items()):
        cap=(16<<20)//(width*count*4);assert cap>=1
        first=last=None
        for y in ys:
            if first is not None and (y!=last+1 or y-first>=cap):
                yield [x,first,width,last-first+1,count];first=None
            if first is None:first=y
            last=y
        if first is not None:yield [x,first,width,last-first+1,count]

def copy_spec(symbol,dtype,box,**extra):
    x,y,w,h,count=box;bits=16 if dtype=='u16' else 32
    assert 0<=x and 0<=y and w>0 and h>0 and x+w<=30 and y+h<=1160
    assert count>0 and w*h*count*4<=16<<20
    return dict(symbol=symbol,dtype=dtype,x=x,y=y,width=w,height=h,count=count,bits=bits,**extra)

def box(item):return [item[k] for k in ('x','y','width','height','count')]
def host_bytes(item):return item['width']*item['height']*item['count']*4
def native_bytes(item):return item['width']*item['height']*item['count']*item['bits']//8

def build(plan,transport,host,typed,observer_limit=128):
    declared=roles(plan,transport);exports={e['name']:e for e in typed['exports']}
    upload=[]
    for category in ('config_uploads','matrix_uploads','parameter_uploads'):
        for original in host[category]:
            e=exports[original['symbol']]
            item=copy_spec(original['symbol'],e['dtype'],[original[k] for k in ('x','y','width','height','count')],
                prepared=dict(original),category=category)
            upload.append(item)
    states=[copy_spec('state','u32',b) for b in rectangles({xy:16 for xy,r in declared.items() if r['role']!=10})]
    transport_copies=[]
    for router,count in ((False,8),(True,32)):
        transport_copies.extend(copy_spec('transport_status','u32',b,router=router)
            for b in rectangles({xy:count for xy,r in declared.items() if (r['role']==10)==router}))
    diagnostic=[]
    for e in typed['exports']:
        if e['name'] in PARAMETERS|CONTROL|{'input'}:continue
        points={}
        for xy,r in declared.items():
            root=r['role']==1 and r['ordinal']==0
            selectors=e['roles']
            active=r['role'] in selectors or ('matrix_root' in selectors and root) or ('linear_matrix_root' in selectors and root and r['matrix_kind']<=4)
            if active:points[xy]=e['count'][str(r['role'])] if isinstance(e['count'],dict) else e['count']
        diagnostic.extend(copy_spec(e['name'],e['dtype'],b) for b in rectangles(points))
    persistent=[copy_spec(p['symbol'],exports[p['symbol']]['dtype'],[p['x'],p['y'],p['width'],p['height'],p['count']])
                for p in host['persistent_exports']]
    hidden=[copy_spec(host[name]['symbol'],'u16',[host[name][k] for k in ('x','y','width','height','count')])
            for name in ('hidden_input','hidden_output')]
    x,y=plan['support']['observer'];observer=copy_spec('observer','u32',[x,y,1,1,32])
    # Static operation families: uploads once, two complete retention reads;
    # state at initialization/three prepared+complete boundaries/reset;
    # persistent state at initialization/reset; two full transport snapshots
    # per completed serial, one diagnostic set per serial, finite observer polls.
    copies=3*len(upload)+8*len(states)+2*len(persistent)+6*len(transport_copies)+3*len(diagnostic)+3+3*observer_limit
    h2d=sum(map(host_bytes,upload))+3*host_bytes(hidden[0])
    d2h=2*sum(map(host_bytes,upload))+8*sum(map(host_bytes,states))+2*sum(map(host_bytes,persistent))+6*sum(map(host_bytes,transport_copies))+3*sum(map(host_bytes,diagnostic))+3*observer_limit*host_bytes(observer)
    raw=8*sum(map(native_bytes,states))+2*sum(map(native_bytes,persistent))+6*sum(map(native_bytes,transport_copies))+3*sum(map(native_bytes,diagnostic))+3*native_bytes(hidden[0])+3*observer_limit*native_bytes(observer)
    files=8*len(states)+2*len(persistent)+6*len(transport_copies)+3*len(diagnostic)+3+3*observer_limit
    return dict(scope='Original complete Layer0 host source for0,1,reset0; no runtime admission',uploads=upload,
        state_copies=states,persistent_copies=persistent,transport_copies=transport_copies,
        diagnostic_copies=diagnostic,hidden_input=hidden[0],hidden_output=hidden[1],observer=observer,
        budget=dict(max_copies=copies,max_H2D_host_bytes=h2d,max_D2H_host_bytes=d2h,max_host_bytes=h2d+d2h,
            max_launches=8,max_observer_polls_per_serial=observer_limit,expected_raw_bytes=raw,expected_raw_files=files,
            failed_copy_raw_reserve_bytes=16<<20,raw_byte_ceiling=raw+(16<<20),raw_file_ceiling=files+1,
            maximum_copy_host_bytes=max(host_bytes(s) for s in upload+states+persistent+transport_copies+diagnostic+[observer]),
            retained_weight_H2D_passes=1,retention_D2H_passes=2,channels=16,blocking_nonweight_copies=True,async_weight_max_tasks=4,
            arithmetic_oracle_calls_during_runtime=0),
        runtime_admitted=False,actual_banks_bound=False,SDK_invocations=0)
