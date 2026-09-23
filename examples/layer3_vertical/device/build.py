"""Original Layer3 vertical placement with disjoint native KV stream regions.

Only geometry, coordinate encoding and physical stream colors change. Logical
tiles, reduction order, numerical kernels, state and protocol identities remain.
This generator performs no SDK operation and reads no model arrays.
"""
from collections import Counter, defaultdict
from pathlib import Path
import copy, hashlib, json

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'layer3-continuous-compile-002'
VERTICAL = HERE.parent / 'layer0_vertical_top'
HEIGHT, BODY, SPINE, FRAME = 1160, 1156, 0, 1
sha = lambda raw: hashlib.sha256(raw).hexdigest()

def vector(name, values):
    return f'const {name}:[{len(values)}]u16=[{len(values)}]u16{{'+','.join(map(str,values))+'};\n'

def build():
    old = json.loads((BASE/'layer3-plan.json').read_bytes())
    prior = json.loads((BASE/'transport-map.json').read_bytes())
    plan = copy.deepcopy(old)
    stripes = plan['stripes']
    kv = tuple(i for i,s in enumerate(stripes) if s['packet_kind'] in (2,3))
    assert len(kv)==16 and sum(s['columns'] for s in stripes)==30576
    native_order = [next(i for i in kv if stripes[i]['packet_kind']==kind and stripes[i]['group']==group)
                    for band in range(4) for kind,group in ((2,2*band),(2,2*band+1),(3,2*band),(3,2*band+1))]
    # Column2 keeps all four streams for each six-head group within its own y
    # interval. Fifteen initial rows keep the first roots below those heads.
    bins=[list(native_order)];used=[15+16*55]
    for i,s in sorted(enumerate(stripes),key=lambda t:(-t[1]['columns'],t[0])):
        if i in kv:continue
        n=s['columns']+1;fits=[(BODY-u-n,j) for j,u in enumerate(used) if u+n<=BODY]
        if fits:j=min(fits)[1]
        else:j=len(bins);bins.append([]);used.append(0)
        bins[j].append(i);used[j]+=n
    # The east residual owner consumes the two extra rows in the final column.
    final=min(range(1,len(bins)),key=lambda j:(used[j],j))
    bins[final],bins[-1]=bins[-1],bins[final];used[final],used[-1]=used[-1],used[final]
    width=len(bins)+2;last=width-1
    assert width==29 and used[-1]+2<=BODY
    cells={};endpoints={};columns=defaultdict(list)
    def mark(point,role,**extra):
        point=tuple(point);assert 0<=point[0]<width and 0<=point[1]<HEIGHT and point not in cells,(point,role)
        cells[point]=dict(role=role,**extra)
    def endpoint(legacy,router,owner,role,**extra):
        key=tuple(legacy);assert key not in endpoints
        assert sum(abs(a-b) for a,b in zip(router,owner))==1
        endpoints[key]=dict(router=list(router),endpoint=list(owner))
        mark(router,9);mark(owner,role,**extra)
        if router[1]!=SPINE:columns[router[0]].append(router[1])
        return list(owner)
    def support(label):
        if label==0:return [0,0],[0,1]
        if label==1:return [1,3],[0,3]
        if label==2:return [last,2],[last,3]
        assert 16<=label<176
        y=4+220*((label-16)//6)+2*((label-16)%6) if label<40 else 84+label-40
        return [1,y],[0,y]
    for key,label,role in [('origin',0,6),('input_norm',1,2),('post_norm',2,3)]:
        r,e=support(label);plan[key]=endpoint([label,0],r,e,role)
    plan['observer']=[0,2];mark(plan['observer'],7)
    heads=[];sinks=[];mlp=[]
    for h in range(24):
        r,e=support(16+h);heads.append(endpoint([16+h,0],r,e,4,head_id=h))
        sink=[e[0],e[1]+1];mark(sink,8,head_id=h);sinks.append(sink)
    for m in range(136):
        r,e=support(40+m);mlp.append(endpoint([40+m,0],r,e,5,index=m))
    for j,indices in enumerate(bins):
        x=j+2;y=17 if j==0 else 4 if x==last else 2
        for logical in indices:
            s=stripes[logical];o=old['stripes'][logical];r=[x,y];owner=[x,y+1]
            if logical in kv:mark(owner,1,ordinal=0,participants=s['columns'],matrix_kind=s['packet_kind'],matrix_group=s['group'])
            else:endpoint([o['x'],o['y']],r,owner,1,ordinal=0,participants=s['columns'],matrix_kind=s['packet_kind'],matrix_group=s['group'])
            for ordinal in range(1,s['columns']):mark([x,y+1+ordinal],1,ordinal=ordinal,participants=s['columns'],matrix_kind=0,matrix_group=0)
            s.update(x=x,y=y+1,router=None if logical in kv else r,axis='y',physical_step=[0,1],
                     logical_stripe=logical,legacy_consumer=o['consumer'],consumer=support(o['consumer'][0])[1])
            y+=s['columns']+1
        assert y<=HEIGHT
    for x in range(width):
        columns[x].sort(reverse=True);assert columns[x] or x==0
        if x:mark([x,0],9)
    assert len(endpoints)==611
    nodes=[];edges=[];segments=[];routes={}
    def route(x,y,color,rx,tx,owner):
        key=(x,y,color);assert key not in routes,(key,owner,routes.get(key))
        assert 0<=x<width and 0<=y<HEIGHT
        routes[key]=dict(rx=rx,tx=tx,owner=owner)
    def wire(a,b,color,owner):
        x,y=a;ex,ey=b;dx=(ex>x)-(ex<x);dy=(ey>y)-(ey<y)
        assert bool(dx)!=bool(dy)
        n=abs(ex-x)+abs(ey-y);out='EAST' if dx>0 else 'WEST' if dx<0 else 'SOUTH' if dy>0 else 'NORTH'
        back=dict(EAST='WEST',WEST='EAST',SOUTH='NORTH',NORTH='SOUTH')[out]
        for i in range(n+1):route(x+i*dx,y+i*dy,color,'RAMP' if i==0 else back,['RAMP'] if i==n else [out],owner)
        segments.append(dict(source=list(a),destination=list(b),color=color,owner=owner))
    for x,ys in sorted(columns.items()):
        for i,y in enumerate(ys):
            neg=(len(ys)-1-i)%2;pos=2 if i==len(ys)-1 else (len(ys)-2-i)%2
            nodes.append(dict(x=x,y=y,spine=False,mask=5|(2 if i else 0),input_colors=[17,7+neg,14+pos],output_colors=[18,14+neg,7+pos]))
            edges.append(dict(child=[x,y],parent=[x,ys[i+1]] if i+1<len(ys) else [x,0],color=7+pos))
        nodes.append(dict(x=x,y=0,spine=True,mask=1|(2 if x else 0)|(4 if x<last else 0),
            input_colors=[17 if x==0 else 9,14+(x-1)%2,7+x%2],output_colors=[18 if x==0 else 16,7+(x-1)%2,14+x%2]))
        if x:edges.append(dict(child=[x,0],parent=[x-1,0],color=7+(x-1)%2))
    for e in edges:
        wire(e['child'],e['parent'],e['color'],'tree_up');wire(e['parent'],e['child'],e['color']+7,'tree_down')
    for e in endpoints.values():
        wire(e['endpoint'],e['router'],17,'endpoint_injection');wire(e['router'],e['endpoint'],18,'endpoint_delivery')
    for s in stripes:
        x,y,n=s['x'],s['y'],s['columns']
        for i in range(n):
            if i<n-1:route(x,y+i,i%2,'SOUTH',['RAMP'],'matrix_reduce_rx')
            if i:route(x,y+i,(i+1)%2,'RAMP',['NORTH'],'matrix_reduce_tx')
            route(x,y+i,2,'RAMP' if i==0 else 'NORTH',['RAMP'] if i==n-1 else ['SOUTH'] if i==0 else ['RAMP','SOUTH'],'matrix_broadcast')
    matrix_ends={x:max(s['y']+s['columns']-1 for s in stripes if s['x']==x) for x in range(2,width)}
    for x in range(width):
        route(x,1,4,'RAMP' if x==0 else 'WEST',(['EAST'] if x<last else [])+(['SOUTH'] if x>=2 else []),'frame_rail')
        if x>=2:
            for y in range(2,matrix_ends[x]+1):
                route(x,y,4,'NORTH',(['RAMP'] if cells.get((x,y),{}).get('role')==1 else [])+(['SOUTH'] if y<matrix_ends[x] else []),'frame_vertical')
    wire(plan['origin'],plan['observer'],5,'origin_observer')
    for h in range(24):wire(heads[h],sinks[h],6,'head_archive')
    native=[];native_routes=[]
    for band in range(4):
        for slot,logical in enumerate(native_order[4*band:4*band+4]):
            s=stripes[logical];x,y=s['x'],s['y'];color=[3,10,11,12][slot];first=4+220*band
            targets=[tuple(p) for p in heads[6*band:6*band+6]];assert x==2 and first+10<y<first+220
            for xx in range(x,-1,-1):
                item=dict(x=xx,y=y,color=color,rx='RAMP' if xx==x else 'EAST',tx=['WEST'] if xx else ['NORTH'],owner='native_KV')
                route(**item);native_routes.append(item)
            for yy in range(y-1,first-1,-1):
                item=dict(x=0,y=yy,color=color,rx='SOUTH',tx=(['RAMP'] if (0,yy) in targets else [])+(['NORTH'] if yy>first else []),owner='native_KV')
                route(**item);native_routes.append(item)
            native.append(dict(logical_stripe=logical,kind=s['packet_kind'],group=s['group'],band=band,color=color,
                source=[x,y],targets=[list(p) for p in targets],physical_region_y=[first,y],head_ids=list(range(6*band,6*band+6))))
    # Every emitted physical edge has a matching receiver of the same color.
    steps=dict(NORTH=(0,-1),SOUTH=(0,1),EAST=(1,0),WEST=(-1,0));opposite=dict(NORTH='SOUTH',SOUTH='NORTH',EAST='WEST',WEST='EAST')
    links=0
    for (x,y,color),v in routes.items():
        for out in v['tx']:
            if out=='RAMP':continue
            dx,dy=steps[out];assert routes[x+dx,y+dy,color]['rx']==opposite[out];links+=1
    traffic=copy.deepcopy(prior['traffic']);parent={tuple(e['child']):tuple(e['parent']) for e in edges};counts=Counter();hops=longest=0
    def ancestors(p):
        values=[p]
        while p in parent:p=parent[p];assert p not in values;values.append(p)
        return values
    for flow in traffic:
        flow['source']=endpoints[tuple(flow['source'])]['router'];flow['destination']=endpoints[tuple(flow['destination'])]['router']
        a=ancestors(tuple(flow['source']));b=ancestors(tuple(flow['destination']));join=next(p for p in a if p in b)
        path=a[:a.index(join)+1]+list(reversed(b[:b.index(join)]));distance=2
        for p,q in zip(path,path[1:]):counts[p,q]+=flow['packets'];distance+=sum(abs(u-v) for u,v in zip(p,q))
        hops+=flow['packets']*distance;longest=max(longest,distance)
    role_counts=Counter(v['role'] for v in cells.values());role_counts[0]=width*HEIGHT-len(cells)
    assert role_counts[1]==30576 and len(nodes)==len(edges)+1==role_counts[9]
    summary=dict(scope='Generated source only; actual compile, fit, routing and numerical qualification still required',
        application=[width,HEIGHT],role_counts=dict(role_counts),matrix_PEs=30576,support_PEs=188,router_PEs=len(nodes),
        idle_PEs=role_counts[0],packet_endpoints=len(endpoints),route_configurations=len(routes),physical_links=links,
        traffic_paths=len(traffic),traffic_packets=prior['summary']['traffic_packets'],packet_hops_including_endpoints=hops,
        longest_packet_path=longest,native_streams=16,native_recipient_deliveries=96,head_archive_paths=24,
        native_stream_colors=[3,10,11,12],native_same_color_regions_disjoint=True,SDK_invocations=0,compiled=False,full_model=False)
    plan.update(scope=summary['scope'],application=[width,HEIGHT],logical_application=old['application'],
        attention_heads=heads,head_observers=sinks,mlp_owners=mlp,logical_matrix_regions=plan.pop('matrix_regions'),
        legacy_weight_batches=plan.pop('weight_batches'),legacy_row_ends=plan.pop('row_ends'),
        physical_placement=dict(matrix_axis='y',ordinal_step=[0,1],reduction='descending logical ordinal north',
            frame_y=1,spine_y=0,matrix_columns=len(bins),native_KV_column=2),
        layer_boundary=dict(input_owner=plan['input_norm'],output_owner=plan['post_norm'],words=5120,dtype='BF16',
            input_face='WEST',output_face='EAST',adjacent_device_handoff_implemented=False),
        original_plan_sha256=sha((BASE/'layer3-plan.json').read_bytes()),compiled=False,SDK_invocations=0)
    transport=dict(summary=summary,nodes=nodes,edges=edges,segments=segments,native_streams=native,native_routes=native_routes,
        endpoints=[dict(legacy=list(k),**v) for k,v in sorted(endpoints.items())],traffic=traffic,
        expected_directed_edge_packets=[dict(source=list(a),destination=list(b),packets=n) for (a,b),n in sorted(counts.items())])

    for p in BASE.glob('*.csl'):
        if p.name not in ('layout.csl','matrix_lane.csl','tree_router.csl','packets.csl','transport_header.csl','native_colors.csl','credit_coordinates.csl','credit_head_coordinates.csl','layer3_pe.csl'):
            (HERE/p.name).write_bytes(p.read_bytes())
    lane=(BASE/'matrix_lane.csl').read_text();assert lane.count('.negative=WEST,.positive=EAST')==1
    (HERE/'matrix_lane.csl').write_text(lane.replace('.negative=WEST,.positive=EAST','.negative=NORTH,.positive=SOUTH'))
    for name in ('packets.csl','transport_header.csl'):(HERE/name).write_bytes((VERTICAL/name).read_bytes())
    router=(BASE/'tree_router.csl').read_text()
    router=router.replace('if(y==config[1]) @as(u16,0) else if(y<config[1])','if(x==config[0]) @as(u16,0) else if(x<config[0])')
    router=router.replace('else if(y==config[1] and x<config[0])','else if(x==config[0] and y>config[1])')
    router=router.replace('config[1]>=45 or (mask!=0 and ((spine and config[0]!=749) or (!spine and config[0]>=749)))',f'config[0]>={width} or config[1]>={HEIGHT} or (mask!=0 and ((spine and config[1]!=0) or (!spine and config[1]==0)))')
    (HERE/'tree_router.csl').write_text(router)
    coords='// Physical support-router coordinates; application labels remain logical.\n'
    coords+=f'fn x(label:u16) u16 {{@assert(label<=2 or (label>=16 and label<176));return if(label==0) @as(u16,0) else if(label==2) @as(u16,{last}) else @as(u16,1);}}\n'
    coords+='fn y(label:u16) u16 {@assert(label<=2 or (label>=16 and label<176));return if(label==0) @as(u16,0) else if(label==1) @as(u16,3) else if(label==2) @as(u16,2) else if(label<40) @as(u16,4)+220*((label-16)/6)+2*((label-16)%6) else @as(u16,84)+label-40;}\n'
    (HERE/'support_coordinates.csl').write_text(coords)
    pe=(BASE/'layer3_pe.csl').read_text();pe=pe.replace('const credit_protocol=', 'const support_coordinates=@import_module("support_coordinates.csl",.{});\nconst credit_protocol=',1)
    for a,b in [
        ('rows.transmit(lane.descriptor[2]+fanout_index,lane.descriptor[3],root_values);','rows.transmit(support_coordinates.x(lane.descriptor[2]+fanout_index),support_coordinates.y(lane.descriptor[2]+fanout_index),root_values);'),
        ('packets.send(request_x,request_y,&tx,8);','packets.send(support_coordinates.x(request_x),support_coordinates.y(request_x),&tx,8);'),
        ('rows.transmit(2,0,&retained_chunk);','rows.transmit(support_coordinates.x(2),support_coordinates.y(2),&retained_chunk);')]:
        assert pe.count(a)==1;pe=pe.replace(a,b)
    (HERE/'layer3_pe.csl').write_text(pe)
    query=sorted((s for s in stripes if s['packet_kind']==1),key=lambda s:s['group']);assert len(query)==96
    (HERE/'credit_coordinates.csl').write_text('// Original Q IDs with physical router coordinates.\n'+vector('root_ids',list(range(96)))+vector('root_x',[s['router'][0] for s in query])+vector('root_y',[s['router'][1] for s in query]))
    text='// Four original Q sources per head; unchanged logical IDs.\nparam head:u16;\ncomptime {@comptime_assert(head<24);}\n'
    for axis in ('x','y'):
        for slot in range(4):text+=vector(f'slot{slot}_{axis}',[query[4*h+slot]['router'][0 if axis=='x' else 1] for h in range(24)])
        text+=f'const source_{axis}:[4]u16=[4]u16{{'+','.join(f'slot{s}_{axis}[head]' for s in range(4))+'};\n'
    (HERE/'credit_head_coordinates.csl').write_text(text)
    color_text='// Four physical colors reused only across PE-disjoint six-head regions.\n'
    for name,values in [('key',[3,10]*4),('value',[11,12]*4)]:color_text+=f'const {name}:[8]color=[8]color{{'+','.join(f'@get_color({c})' for c in values)+'};\n'
    color_text+='fn head_streams(head:u16) [4]color {const g:u16=2*(head/6);return [4]color{key[g],key[g+1],value[g],value[g+1]};}\n'
    (HERE/'native_colors.csl').write_text(color_text)
    physical=sorted(stripes,key=lambda s:(s['x'],s['y']));starts=[0];rstarts=[0];routers=[]
    for x in range(width):starts.append(starts[-1]+sum(s['x']==x for s in physical));routers.extend(columns[x]);rstarts.append(len(routers))
    layout=f'// Generated original Layer3 vertical source.\nconst memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={HEIGHT}}});\n'
    for name,values in [('starts',starts),('sy',[s['y'] for s in physical]),('lengths',[s['columns'] for s in physical]),('kinds',[s['packet_kind'] for s in physical]),('groups',[s['group'] for s in physical]),('router_y',routers),('router_start',rstarts)]:layout+=vector(name,values)
    layout+=f'layout {{\n @set_rectangle({width},{HEIGHT});\n for(@range(u16,{width}))|x|{{for(@range(u16,{HEIGHT}))|y|{{\n'
    layout+='''  var role:u16=0;var ordinal:u16=0;var participants:u16=54;var kind:u16=0;var group:u16=0;var head:u16=0;
  if(x==0){if(y==1){role=6;}else if(y==2){role=7;}else if(y==3){role=2;}
   else if(y>=84 and y<220){role=5;}
   else if(y>=4 and y<676){const band:u16=(y-4)/220;const dy:u16=(y-4)%220;
    if(dy<12){role=if(dy%2==0) @as(u16,4) else @as(u16,8);head=6*band+dy/2;}}}
'''
    layout+=f'  if(x=={last} and y==3){{role=3;}}\n'
    layout+='''  for(@range(u16,starts[x+1]-starts[x]))|j|{const s=starts[x]+j;
   if(y>=sy[s] and y<sy[s]+lengths[s]){role=1;ordinal=y-sy[s];participants=lengths[s];
    if(ordinal==0){kind=kinds[s];if(kind==2 or kind==3){group=groups[s];}}}}
  var index:u16=65535;
  for(@range(u16,router_start[x+1]-router_start[x]))|i|{if(router_y[router_start[x]+i]==y){index=i;}}
  if(index!=65535 or y==0){
   var mask:u16=0;var ins=[3]u16{0,0,0};var outs=[3]u16{0,0,0};
'''
    layout+=f'   if(y==0){{mask=1|(if(x>0) @as(u16,2) else @as(u16,0))|(if(x<{last}) @as(u16,4) else @as(u16,0));\n'
    layout+='''    ins=[3]u16{if(x==0) @as(u16,17) else @as(u16,9),14+(x+1)%2,7+x%2};outs=[3]u16{if(x==0) @as(u16,18) else @as(u16,16),7+(x+1)%2,14+x%2};
   }else{const count=router_start[x+1]-router_start[x];const negative=(count-1-index)%2;const positive:u16=if(index==count-1) 2 else (count-2-index)%2;
    mask=5|(if(index>0) @as(u16,2) else @as(u16,0));ins=[3]u16{17,7+negative,14+positive};outs=[3]u16{18,14+negative,7+positive};}
   @set_tile_code(x,y,"tree_router.csl",.{.memcpy_params=memcpy.get_params(x),.spine=y==0,.mask=mask,.input_colors=ins,.output_colors=outs});
  }else{@set_tile_code(x,y,"layer3_pe.csl",.{.memcpy_params=memcpy.get_params(x),.role=role,.ordinal=ordinal,.participants=participants,.matrix_kind=kind,.matrix_group=group,.head_id=head});}
 }}
'''
    for seg in segments:
        x,y=seg['source'];ex,ey=seg['destination'];dx=(ex>x)-(ex<x);dy=(ey>y)-(ey<y);n=abs(ex-x)+abs(ey-y)
        out='EAST' if dx>0 else 'WEST' if dx<0 else 'SOUTH' if dy>0 else 'NORTH';back=opposite[out]
        xx=f'{x}+i' if dx>0 else f'{x}-i' if dx<0 else str(x);yy=f'{y}+i' if dy>0 else f'{y}-i' if dy<0 else str(y)
        layout+=f' for(@range(u16,{n+1}))|i|{{@set_color_config({xx},{yy},@get_color({seg["color"]}),.{{.routes=.{{.rx=if(i==0) .{{RAMP}} else .{{{back}}},.tx=if(i=={n}) .{{RAMP}} else .{{{out}}}}}}});}}\n'
    for x in range(width):
        tx=(['EAST'] if x<last else [])+(['SOUTH'] if x>=2 else [])
        layout+=f' @set_color_config({x},1,@get_color(4),.{{.routes=.{{.rx=.{{{"RAMP" if x==0 else "WEST"}}},.tx=.{{{",".join(tx)}}}}}}});\n'
        if x<2:continue
        stop=matrix_ends[x]
        layout+=f' for(@range(u16,{stop-1}))|i|{{const y:u16=2+i;var matrix:bool=false;var ordinal:u16=0;\n'
        layout+=f'  for(@range(u16,starts[{x+1}]-starts[{x}]))|k|{{const s=starts[{x}]+k;if(y>=sy[s] and y<sy[s]+lengths[s]){{matrix=true;ordinal=y-sy[s];}}}}\n'
        layout+=f'  if(matrix){{@set_color_config({x},y,@get_color(4),.{{.routes=.{{.rx=.{{NORTH}},.tx=if(y=={stop}) .{{RAMP}} else .{{RAMP,SOUTH}}}},.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter=(17472-ordinal*96)%17472,.max_counter=95,.limit1=17471}}}});}}\n'
        layout+=f'  else{{@set_color_config({x},y,@get_color(4),.{{.routes=.{{.rx=.{{NORTH}},.tx=.{{SOUTH}}}}}});}}\n }}\n'
    for item in native_routes:
        layout+=f' @set_color_config({item["x"]},{item["y"]},@get_color({item["color"]}),.{{.routes=.{{.rx=.{{{item["rx"]}}},.tx=.{{{",".join(item["tx"])}}}}}}});\n'
    exports=[line for line in (BASE/'layout.csl').read_text().splitlines() if '@export_name(' in line]
    layout+='\n'.join(exports)+'\n}\n';(HERE/'layout.csl').write_text(layout)
    reuse={p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()),exact_accepted_Layer3=(BASE/p.name).exists() and (BASE/p.name).read_bytes()==p.read_bytes(),exact_accepted_Layer0=(VERTICAL/p.name).exists() and (VERTICAL/p.name).read_bytes()==p.read_bytes()) for p in HERE.glob('*.csl')}
    for name,value in [('plan.json',plan),('transport-map.json',transport),('SOURCE-REVIEW.json',dict(summary=summary,source_files=reuse))]:
        (HERE/name).write_text(json.dumps(value,indent=2 if name!='transport-map.json' else None)+'\n')
    (HERE/'logical-plan.json').write_bytes((BASE/'layer3-plan.json').read_bytes())
    print(json.dumps(summary))

if __name__=='__main__':build()
