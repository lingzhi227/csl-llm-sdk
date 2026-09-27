"""Restricted spatial group128 maximum, scale broadcast and ordered gather."""
from dataclasses import asdict,dataclass

@dataclass(frozen=True)
class SpatialQuantPlan:
    width:int=8
    height:int=4
    max_row_colors:tuple=(3,4)
    max_column_colors:tuple=(5,6)
    scale_color:int=7
    gather_row_colors:tuple=(8,9)
    gather_column_colors:tuple=(10,11)
    input_queues:tuple=(2,3,4,5,6)
    output_queues:tuple=(2,3,4)
    local_tasks:tuple=(8,9,10,11,12)
    dsrs:tuple=(3,4,5,6,7)

    @property
    def items(self):return 128//(self.width*self.height)
    def children(self,x,y):return ([(x+1,y)] if x+1<self.width else [])+([(0,y+1)] if x==0 and y+1<self.height else [])
    def subtree(self,x,y):
        return [yy*self.width+xx for yy in range(y,self.height) for xx in range(self.width)] if x==0 else [y*self.width+xx for xx in range(x,self.width)]
    def routes(self):
        rows=[]
        def add(x,y,c,rx,tx):rows.append(dict(x=x,y=y,color=c,rx=rx,tx=tx))
        for y in range(self.height):
            for x in range(self.width):
                add(x,y,self.scale_color,'RAMP' if x==y==0 else 'WEST' if x else 'NORTH',([] if x==y==0 else ['RAMP'])+(['EAST'] if x+1<self.width else [])+(['SOUTH'] if x==0 and y+1<self.height else []))
                for rc,cc in [(self.max_row_colors,self.max_column_colors),(self.gather_row_colors,self.gather_column_colors)]:
                    for cx,cy in self.children(x,y):add(x,y,rc[cx%2] if cx else cc[cy%2],'EAST' if cx else 'SOUTH',['RAMP'])
                    if x or y:add(x,y,rc[x%2] if x else cc[y%2],'RAMP',['WEST' if x else 'NORTH'])
        return rows
    def validate(self):
        n=self.width*self.height
        if not (2<=self.width<=16 and 2<=self.height<=16 and n in (4,8,16,32,64)):raise ValueError('Even local partition of exactly128 inputs required')
        colors=self.max_row_colors+self.max_column_colors+(self.scale_color,)+self.gather_row_colors+self.gather_column_colors
        if len(colors)!=9 or len(set(colors))!=9 or any(not 2<=c<=20 for c in colors):raise ValueError('Reserved/overlapping colors')
        for ids,length,low,high in [(self.input_queues,5,2,7),(self.output_queues,3,2,7),(self.local_tasks,5,8,20),(self.dsrs,5,3,7)]:
            if len(ids)!=length or len(set(ids))!=length or any(not low<=i<=high for i in ids):raise ValueError('Resource collision')
        routes=self.routes();keyed={(r['x'],r['y'],r['color']):r for r in routes}
        if len(keyed)!=len(routes):raise ValueError('Duplicate route')
        steps={'EAST':(1,0,'WEST'),'WEST':(-1,0,'EAST'),'SOUTH':(0,1,'NORTH'),'NORTH':(0,-1,'SOUTH')};hops=0
        for r in routes:
            for d in r['tx']:
                if d=='RAMP':continue
                dx,dy,rx=steps[d];dest=keyed.get((r['x']+dx,r['y']+dy,r['color']))
                if dest is None or dest['rx']!=rx:raise ValueError('Disconnected link')
                hops+=1
        if hops!=3*(n-1):raise ValueError('Missing maximum/scale/gather tree edge')
        for y in range(self.height):
            for x in range(self.width):
                ordered=[y*self.width+x]
                for cx,cy in self.children(x,y):ordered+=self.subtree(cx,cy)
                if ordered!=self.subtree(x,y) or len(set(ordered))!=len(ordered):raise ValueError('Overlapping/misordered gather slices')
        return dict(pes=n,items_per_pe=self.items,group_size=128,root_operand_words=65,tree_links=hops,full_model=False,consumer='one original2x128 FP8 tile on root, no host transfer between producer and dot',compiled_sram_admitted=False,
            stack_allowance=4096,sram_ceiling=48128,protocol='Arm all receives; ready audit before start; maximum join, one scale broadcast, ordered encoded-payload join, send-completion credits',
            timing='Same-root start through full65-word packet and a separate native-dot interval, after explicit receive arming. Host arm/readiness excluded and reported separately; launch-entry skew included.')
    def document(self):return dict(schema='wse-spatial-quant-v1',plan=asdict(self),resources=self.validate(),routes=self.routes())
    def emit_layout(self):
        self.validate();w,h=self.width,self.height
        lines=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={w},.height={h}}});','layout {',f'  @set_rectangle({w},{h});']
        for y in range(h):
            for x in range(w):
                p=dict(x=x,y=y,width=w,height=h,items=self.items,
                    max_east_color=self.max_row_colors[(x+1)%2],max_south_color=self.max_column_colors[(y+1)%2],max_out_color=self.max_row_colors[x%2] if x else self.max_column_colors[y%2],scale_color=self.scale_color,
                    gather_east_color=self.gather_row_colors[(x+1)%2],gather_south_color=self.gather_column_colors[(y+1)%2],gather_out_color=self.gather_row_colors[x%2] if x else self.gather_column_colors[y%2])
                p.update(dict(zip(('max_east_iq','max_south_iq','scale_iq','gather_east_iq','gather_south_iq'),self.input_queues)))
                p.update(dict(zip(('max_oq','scale_oq','gather_oq'),self.output_queues)))
                p.update(dict(zip(('max_sent_task','scale_task','east_task','south_task','gather_sent_task'),self.local_tasks)))
                p.update(dict(zip(('max_dsr','scale_dsr','east_dsr','south_dsr','gather_dsr'),self.dsrs)))
                lines.append('  @set_tile_code(%d,%d,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),%s});'%(x,y,x,','.join('.%s=%d'%(k,v) for k,v in p.items())))
        for r in self.routes():lines.append('  @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});'%(r['x'],r['y'],r['color'],r['rx'],','.join(r['tx'])))
        for name,ty in [('input','f32'),('codes','u16'),('scale','u32'),('packet','u32'),('audit','u32'),('ticks','u16'),('weights','u16'),('weight_scale','f32'),('product','f32'),('prepared','u32')]:lines.append(f'  @export_name("{name}",[*]{ty},false);')
        lines+=['  @export_name("prepare_weight",fn()void);','  @export_name("arm",fn()void);','  @export_name("start",fn()void);','}'];return '\n'.join(lines)+'\n'
