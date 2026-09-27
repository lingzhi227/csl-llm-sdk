"""WSE projection lowering: partition -> streams/completions -> routes/resources.

The first backend is orthogonal K-input/M-output regions. It preserves the native
kernel's tile and addition order while making physical placement a parameter.
Completion edges describe completed streams, not extra runtime global barriers.
"""
from dataclasses import asdict, dataclass
from graphlib import TopologicalSorter
from .contraction import tree


@dataclass(frozen=True)
class ProjectionSpec:
    output_rows: int = 48
    input_columns: int = 5120
    tile_rows: int = 2
    tile_columns: int = 128
    k_axis: str = 'x'
    max_input_hops: int = 80
    max_link_words: int = 128
    max_contribution_transport_hops: int = 160


class ProjectionFlow:
    def __init__(self, spec=ProjectionSpec()):
        self.spec=spec
        if spec.k_axis not in ['x','y']:raise ValueError('K axis')
        if spec.tile_rows*spec.tile_columns!=256 or spec.tile_rows not in [2,4,8]:raise ValueError('Qualified native tile family')
        if spec.output_rows%spec.tile_rows or spec.input_columns%spec.tile_columns:raise ValueError('Exact original partition required')
        self.groups=spec.output_rows//spec.tile_rows;self.k=spec.input_columns//spec.tile_columns
        if min(self.groups,self.k)<1:raise ValueError('Empty projection')
        self.rows=spec.tile_rows;self.columns=spec.tile_columns;self.packet_words=self.columns//2+1
        self.nodes=tree(self.k);self.gather_first=max(self.node_color(n) for n in self.nodes[1:])+1
        self.gather_levels=(self.groups-1).bit_length();self.kick_color=self.gather_first+self.gather_levels;self.return_color=self.kick_color+1
        if self.return_color>=20:raise ValueError('SDK/unqualified router color required')
        self.width,self.height=self.xy(self.k+1,self.groups+1)
        if self.width>750 or self.height>1160:raise ValueError('Application geometry')

    def xy(self,k,m):return (k,m) if self.spec.k_axis=='x' else (m,k)
    def worker_xy(self,group,k):return self.xy(k+1,group+1)
    def producer_xy(self,k):return self.xy(k+1,0)
    def node_color(self,n):return 3+2*(n['depth']-1)+n['side']

    def gather(self,group):
        steps=0
        while group%(1<<(steps+1))==0 and group+(1<<steps)<self.groups:steps+=1
        return dict(steps=steps,words=self.rows*min(1<<steps,self.groups-group),
                    color=self.return_color if group==0 else self.gather_first+(group&-group).bit_length()-1)

    def streams(self):
        streams=[dict(id='kick',kind='kick',color=self.kick_color,words=1,control_words=0,
                      path=[self.xy(k,0) for k in range(self.k+1)],consumers=[self.producer_xy(k) for k in range(self.k)])]
        for k in range(self.k):
            streams.append(dict(id=f'input:{k}',kind='input',color=2,words=self.packet_words,control_words=1,
                                path=[self.xy(k+1,g) for g in range(self.groups+1)],consumers=[self.worker_xy(g,k) for g in range(self.groups)]))
        for group in range(self.groups):
            for n in self.nodes[1:]:
                streams.append(dict(id=f'native:{group*self.k+n["rank"]}',kind='native',color=self.node_color(n),words=self.rows+1,control_words=0,
                                    path=[self.worker_xy(group,k) for k in range(n['rank'],n['parent']-1,-1)],consumers=[self.worker_xy(group,n['parent'])]))
            if group:
                parent=group-(group&-group)
                streams.append(dict(id=f'gather:{group}',kind='gather',color=self.gather(group)['color'],words=self.gather(group)['words'],control_words=0,
                                    path=[self.worker_xy(g,0) for g in range(group,parent-1,-1)],consumers=[self.worker_xy(parent,0)]))
        streams.append(dict(id='delivery',kind='delivery',color=self.return_color,words=self.spec.output_rows,control_words=0,
                            path=[self.xy(1,1),self.xy(1,0),self.xy(0,0)],consumers=[self.xy(0,0)]))
        return streams

    @staticmethod
    def direction(a,b):return {(1,0):'EAST',(-1,0):'WEST',(0,1):'SOUTH',(0,-1):'NORTH'}[(b[0]-a[0],b[1]-a[1])]

    def routes(self):
        routes={}
        for stream in self.streams():
            path=stream['path'];consumers=set(stream['consumers'])
            for i,xy in enumerate(path):
                key=(*xy,stream['color'])
                if key in routes:raise ValueError('Concurrent PE/color stream collision: '+str(key))
                tx=(['RAMP'] if xy in consumers else [])+([self.direction(xy,path[i+1])] if i+1<len(path) else [])
                if not tx:raise ValueError('Stream dead end')
                routes[key]=dict(xy=list(xy),color=stream['color'],rx='RAMP' if i==0 else self.direction(xy,path[i-1]),tx=tx,
                                 teardown=stream['kind']=='input',filtered=False,stream=stream['id'])
        return list(routes.values())

    def completion_dag(self):
        events={'inputs-ready':[], 'kick':['inputs-ready']}
        for k in range(self.k):events[f'input:{k}']=['kick']
        for g in range(self.groups):
            for n in self.nodes:
                events[f'sum:{g*self.k+n["rank"]}']=[f'input:{n["rank"]}']+[f'sum:{g*self.k+c}' for c in n['children']]
            events[f'gather:{g}']=[f'sum:{g*self.k}']+[f'gather:{g+2**s}' for s in range(self.gather(g)['steps'])]
        events['delivered']=['gather:0']
        list(TopologicalSorter(events).static_order())
        return events

    def cost(self):
        links={};input_hops=0;word_hops={};streams=self.streams()
        for stream in streams:
            path=stream['path'];words=stream['words']+stream['control_words']
            word_hops[stream['kind']]=word_hops.get(stream['kind'],0)+words*(len(path)-1)
            if stream['kind']=='input':input_hops=max(input_hops,len(path)-1)
            for a,b in zip(path,path[1:]):
                key=(*a,*b);links[key]=links.get(key,0)+words
        # For any native leaf: kick to K owner + vertical input + horizontal
        # native tree (telescoping K coordinates) + vertical gather + delivery.
        causal=max((k+1)+(g+1)+k+g+2 for g in range(self.groups) for k in range(self.k))
        maximum=max(links.values());violations=[]
        if input_hops>self.spec.max_input_hops:violations.append('input path')
        if maximum>self.spec.max_link_words:violations.append('directed-link traffic')
        if causal>self.spec.max_contribution_transport_hops:violations.append('single-contribution transport path')
        return dict(directed_links=len(links),max_link_words_including_teardown=maximum,
                    busiest_links=[dict(source=list(key[:2]),destination=list(key[2:]),words=value) for key,value in links.items() if value==maximum],
                    link_loads=[dict(source=list(key[:2]),destination=list(key[2:]),words=value) for key,value in sorted(links.items())],
                    word_hops_by_kind=word_hops,max_input_hops=input_hops,max_contribution_transport_hops=causal,
                    input_parallel_streams=self.k,active_workers=self.groups*self.k,macs=self.spec.output_rows*self.spec.input_columns,
                    critical_timing_unmeasured=True,full_credit_critical_path_modeled=False,root_gather_chunks=self.groups,violations=violations,
                    scope='Static per-epoch directed-link traffic and single input/native/gather contribution-path hop accounting. Includes one teardown word per input stream. Does not model contention timing, callback cost, queue occupancy or predict tokens/s.')

    def document(self):
        routes=self.routes();cost=self.cost();events=self.completion_dag()
        if cost['violations']:raise ValueError('Static cost admission: '+', '.join(cost['violations']))
        return dict(schema='wse-projection-flow-v1',spec=asdict(self.spec),application=[self.width,self.height],
                    partitions=dict(worker_grid=[self.k,self.groups],k_axis=self.spec.k_axis,local_reduction='ordered binary K tree within each output row region',cross_region='ordered chunked output gather along M axis'),
                    input_owners=[dict(k=k,xy=list(self.producer_xy(k)),columns=[k*self.columns,(k+1)*self.columns],packet_words=self.packet_words) for k in range(self.k)],
                    output_consumer=dict(xy=[0,0],rows=self.spec.output_rows,kind='qualification sink; future lowering must bind a real adjacent consumer'),
                    workers=[dict(rank=g*self.k+k,group=g,k=k,xy=list(self.worker_xy(g,k)),row_start=g*self.rows,column_start=k*self.columns) for g in range(self.groups) for k in range(self.k)],
                    streams=self.streams(),completion_dag=events,routes=routes,cost=cost,
                    buffers=[dict(name='packet',owner='input producer and each worker',bytes=4*self.packet_words,release='producer send completion / worker native completion'),
                             dict(name='child-left/right',owner='worker',bytes_each=4*(self.rows+1),release='both callbacks and joined sum consumed'),
                             dict(name='decoded/sum',owner='worker',bytes=512,release='native complete before ordered join; outgoing send complete before reuse'),
                             dict(name='gather-chunk',owner='root',bytes=4*self.rows,release='send callback before next receive')],
                    resources=dict(input_queue=2,left_queue=3,right_queue=4,gather_queue=5,output_queue=2,receive_ut=2,left_ut=3,right_ut=4,send_ut=5,gather_ut=6,
                                   input_dsr=2,left_dsr=3,right_dsr=5,gather_dsr=1,send_dsr=6,native_dsr=4,bf16_dsr=7,
                                   sdk_reserved_dsrs=[0],sdk_reserved_queues_uts=[0,1],sdk_reserved_colors=[20,21,22,23]),
                    protocol='All receivers arm before kick. Kick arrival starts independent K input producers. Worker input arrival executes native arithmetic; two child callbacks gate ordered local-left-right join. Root sends bounded chunks; send completion credits the next gather receive. Input teardown and local finish gate the next epoch.',
                    compiled_sram_qualified=False,physical=False,full_model=False)

    def emit_layout(self, capacity_workers):
        """Bind qualified native bank capacities to the new component coordinates.

        Capacity provenance comes from the original fixture. It is not a claim
        that these compact coordinates already belong to the global model atlas.
        """
        self.document()
        if len(capacity_workers)!=self.k*self.groups:raise ValueError('Capacity population')
        workers={self.worker_xy(g,k):capacity_workers[g*self.k+k] for g in range(self.groups) for k in range(self.k)}
        producer={self.producer_xy(k):k for k in range(self.k)}
        lines=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={self.width},.height={self.height}}});',
               'layout {',f' @set_rectangle({self.width},{self.height});']
        for y in range(self.height):
            for x in range(self.width):
                xy=(x,y);w=workers.get(xy);g,k=(w['rank']//self.k,w['rank']%self.k) if w else (0,0)
                n=self.nodes[k];children=n['children'] if w else [];gather=self.gather(g) if w and k==0 else dict(steps=0,words=1,color=self.return_color)
                params=dict(role=0 if xy==(0,0) else 1 if w else 3 if xy in producer else 2,
                            rows=self.rows,columns=self.columns,total_rows=self.spec.output_rows,k_blocks=self.k,gather_first=self.gather_first,
                            kick_color=self.kick_color,return_color=self.return_color,
                            fp8_slots=w['fp8_slots'] if w else 0,bf16_slots=w['bf16_slots'] if w else 0,k_rank=k,
                            children=len(children),subtree=n['size'] if w else 0,
                            left_color=self.node_color(self.nodes[children[0]]) if children else 3,
                            right_color=self.node_color(self.nodes[children[1]]) if len(children)>1 else 4,
                            out_color=self.node_color(n) if w and k else gather['color'],gather_steps=gather['steps'],gather_words=gather['words'])
                args=','.join('.%s=%s'%(name,str(value).lower()) for name,value in params.items())
                lines.append(f' @set_tile_code({x},{y},"matrix_epoch.csl",.{{.memcpy_params=memcpy.get_params({x}),{args}}});')
        for route in self.routes():
            x,y=route['xy'];extra=',.teardown=true' if route['teardown'] else ''
            lines.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}%s});'%(x,y,route['color'],route['rx'],','.join(route['tx']),extra))
        for name,dtype in [('bank','u32'),('target','u32'),('packet','u32'),('control','u16'),('local_result','f32'),('aggregate','u32'),('gather','u32'),('audit','u16'),('ticks','u16')]:
            lines.append(' @export_name("%s",[*]%s,%s);'%(name,dtype,str(name=='target').lower()))
        for name,signature in [('initialize_aliases','fn()void'),('arm','fn()void'),('start','fn()void'),('check_payload','fn(u16)void')]:
            lines.append(' @export_name("%s",%s);'%(name,signature))
        lines.append('}')
        return '\n'.join(lines)+'\n'
