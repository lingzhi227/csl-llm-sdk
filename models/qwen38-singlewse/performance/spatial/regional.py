"""Restricted regional GEMV routes, resources and deterministic CSL lowering.

Column multicast, ascending-K row sums, row-result return and column0 completion.
This is a component plan, not full-model placement or a general deadlock proof.
"""
from dataclasses import asdict,dataclass

@dataclass(frozen=True)
class RegionalPlan:
    width:int=8
    height:int=8
    operand_color:int=3
    sum_colors:tuple=(4,5)
    return_color:int=6
    ack_colors:tuple=(7,8)
    input_queues:tuple=(2,3,4,5)
    output_queues:tuple=(2,3,4)
    tasks:tuple=(8,9,10,11,12,13)
    # Compiler/memcpy use0:2. Row-return and sum receives are mutually exclusive.
    dsrs:tuple=(3,4,5,6,7)

    def routes(self):
        routes=[]
        def add(x,y,c,rx,tx):routes.append(dict(x=x,y=y,color=c,rx=rx,tx=tx))
        for y in range(self.height):
            for x in range(self.width):
                add(x,y,self.operand_color,'RAMP' if y==0 else 'NORTH',(['RAMP'] if y else [])+(['SOUTH'] if y+1<self.height else []))
                if x+1<self.width:add(x,y,self.sum_colors[x%2],'RAMP',['EAST'])
                if x>0:add(x,y,self.sum_colors[(x-1)%2],'WEST',['RAMP'])
                add(x,y,self.return_color,'RAMP' if x+1==self.width else 'EAST',['RAMP'] if x==0 else ['WEST'])
                if x==0:
                    if y>0:add(x,y,self.ack_colors[y%2],'RAMP',['NORTH'])
                    if y+1<self.height:add(x,y,self.ack_colors[(y+1)%2],'SOUTH',['RAMP'])
        return routes

    def validate(self):
        if not (2<=self.width<=40 and 2<=self.height<=32):raise ValueError('Bounded regional component geometry')
        colors=(self.operand_color,)+self.sum_colors+(self.return_color,)+self.ack_colors
        if len(self.sum_colors)!=2 or len(self.ack_colors)!=2 or len(set(colors))!=6 or any(not 2<=c<=20 for c in colors):raise ValueError('Color collision or reservation')
        for ids,n,low,high in [(self.input_queues,4,2,7),(self.output_queues,3,2,7),(self.tasks,6,8,20),(self.dsrs,5,3,7)]:
            if len(ids)!=n or len(set(ids))!=n or any(not low<=i<=high for i in ids):raise ValueError('Resource collision or reservation')
        if self.dsrs[1]!=4:raise ValueError('Native dot module leases DSR4')
        keyed={};hops=0
        for r in self.routes():
            key=(r['x'],r['y'],r['color'])
            if key in keyed:raise ValueError('Overlapping route on same PE/color')
            keyed[key]=r
        steps={'EAST':(1,0,'WEST'),'WEST':(-1,0,'EAST'),'SOUTH':(0,1,'NORTH'),'NORTH':(0,-1,'SOUTH')}
        for r in keyed.values():
            for d in r['tx']:
                if d=='RAMP':continue
                dx,dy,rx=steps[d];dest=keyed.get((r['x']+dx,r['y']+dy,r['color']))
                if dest is None or dest['rx']!=rx:raise ValueError('Unmatched route')
                hops+=1
        expected=self.width*(self.height-1)+2*self.height*(self.width-1)+self.height-1
        if hops!=expected:raise ValueError('Missing plane link')
        return dict(passed=True,pes=self.width*self.height,physical_links_across_planes=hops,
            input_words=65,weight_rows=2,k_block=128,source_quantization=False,
            live_bytes_estimate=2048,code_allowance=8192,stack_allowance=4096,sram_ceiling=48128,
            compiled_sram_admitted=False,full_model=False,
            dsr_lifetimes={str(self.dsrs[0]):'operand receive',str(self.dsrs[1]):'dot after operand complete',str(self.dsrs[2]):'sum receive on x>0 OR final row receive on x=0',str(self.dsrs[3]):'column producer transmit',str(self.dsrs[4]):'partial/result send, then completion ack only after send callback'},
            completion='sum ready AND local dot ready; source packet and local result sends must complete before buffer reuse; root joins all row results',
            limitations=['Fixed restricted geometry/routes','One host-launched region epoch; launch-entry skew included','No full-model bank placement or general deadlock proof'])

    def document(self):return dict(schema='wse-regional-gemv-v1',plan=asdict(self),resources=self.validate(),routes=self.routes())

    def emit_layout(self):
        self.validate();w,h=self.width,self.height
        out=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={w},.height={h}}});','layout {',f'  @set_rectangle({w},{h});']
        for y in range(h):
            for x in range(w):
                p=dict(x=x,y=y,width=w,height=h,operand_color=self.operand_color,sum_out_color=self.return_color if x==w-1 else self.sum_colors[x%2],sum_in_color=self.sum_colors[(x-1)%2],return_color=self.return_color,ack_out_color=self.ack_colors[y%2],ack_in_color=self.ack_colors[(y+1)%2])
                p.update(dict(zip(('operand_iq','sum_iq','row_iq','ack_iq'),self.input_queues)))
                p.update(dict(zip(('operand_oq','sum_oq','ack_oq'),self.output_queues)))
                p.update(dict(zip(('operand_recv_task','operand_send_task','sum_recv_task','sum_send_task','row_recv_task','ack_send_task'),self.tasks)))
                p.update(dict(zip(('operand_dsr','dot_dsr','sum_recv_dsr','operand_send_dsr','sum_send_dsr'),self.dsrs)))
                out.append('  @set_tile_code(%d,%d,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),%s});'%(x,y,x,','.join('.%s=%d'%(k,v) for k,v in p.items())))
        for r in self.routes():out.append('  @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});'%(r['x'],r['y'],r['color'],r['rx'],','.join(r['tx'])))
        for n,t in [('weights','u16'),('raw','u32'),('packet','u32'),('scale','f32'),('output','f32'),('row_result','f32'),('ticks','u16'),('audit','u32')]:out.append(f'  @export_name("{n}",[*]{t},false);')
        out+=['  @export_name("execute",fn(u16)void);','}'];return '\n'.join(out)+'\n'
