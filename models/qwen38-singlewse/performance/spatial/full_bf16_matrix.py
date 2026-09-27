"""One complete original matrix at its frozen columnar tile coordinates.

The qualification source occupies an adjacent PE temporarily. Other global
roles are not admitted. This emits an executable slice, not a full-model layout.
"""
import bisect
import hashlib
import json
from .contraction import tree


class FullBf16MatrixPlan:
    def __init__(self, base_path, overlay_path, diagnostic=False):
        self.diagnostic=diagnostic
        self.base=json.loads(base_path.read_text());self.overlay=json.loads(overlay_path.read_text())
        assert hashlib.sha256(base_path.read_bytes()).hexdigest()==self.overlay['base_atlas_sha256']
        self.provenance=dict(base_atlas_sha256=self.overlay['base_atlas_sha256'],overlay_sha256=hashlib.sha256(overlay_path.read_bytes()).hexdigest())
        self.matrix=next(m for m in self.overlay['matrices'] if m['tensor']=='model.language_model.layers.0.linear_attn.in_proj_a.weight')
        assert self.matrix['shape']==[48,5120] and self.matrix['k_blocks']==40
        assert len(self.matrix['segments'])==1
        self.k=40;self.groups=24;self.nodes=tree(40)
        segments=self.base['bf16_eligible_segments'];starts=[s['compact_start'] for s in segments]
        start=(self.matrix['stream_start']+self.overlay['classes']['bf16']['phase'])%824000
        self.workers=[]
        for rank in range(960):
            eligible=start+rank;s=segments[bisect.bisect_right(starts,eligible)-1]
            bank=s['bank_start']+eligible-s['compact_start'];row,col=divmod(bank,750)
            x=col if row%2==0 else 749-col;y=self.base['geometry']['bank_rows'][row]
            def slots(key,owner):
                c=self.overlay['classes'][key];ordinal=(owner-c['phase'])%c['pes']
                return max(0,(c['stream_span']+c['pes']-1-ordinal)//c['pes'])
            fp=slots('fp8-main',bank)
            if row<1146 and 1<=x<=748:fp+=slots('fp8-wide',row*748+(x-1 if row%2==0 else 748-x))
            bf=slots('bf16',eligible)
            assert fp<=111 and bf<=13 and fp*260+bf*512<=35256
            self.workers.append(dict(rank=rank,group=rank//40,k=rank%40,global_xy=[x,y],eligible_rank=eligible,bank_rank=bank,fp8_slots=fp,bf16_slots=bf,target_slot=self.matrix['segments'][0]['slot']))
        assert all(abs(a['global_xy'][0]-b['global_xy'][0])+abs(a['global_xy'][1]-b['global_xy'][1])==1 for a,b in zip(self.workers,self.workers[1:]))
        self.origin=[min(w['global_xy'][0] for w in self.workers)-1,min(w['global_xy'][1] for w in self.workers)]
        assert self.origin==[119,705]
        self.width=631;self.height=2
        for w in self.workers:w['xy']=[w['global_xy'][0]-self.origin[0],w['global_xy'][1]-self.origin[1]]
        self.byxy={tuple(w['xy']):w for w in self.workers}
        self.input_path=[(x,0) for x in range(self.width)]+[(x,1) for x in range(self.width-1,0,-1)]

    def gather(self,group):
        steps=0
        while group%(1<<(steps+1))==0 and group+(1<<steps)<self.groups:steps+=1
        size=min(1<<steps,self.groups-group)
        return dict(steps=steps,words=2*size,color=18 if group==0 else 12+((group&-group).bit_length()-1))

    def routes(self):
        routes={}
        def direction(a,b):return {(1,0):'EAST',(-1,0):'WEST',(0,1):'SOUTH',(0,-1):'NORTH'}[(b[0]-a[0],b[1]-a[1])]
        def add(xy,color,rx,tx,filtered=False,teardown=False):
            key=(*xy,color)
            if key in routes:raise ValueError('PE/color collision: '+str(key))
            routes[key]=dict(xy=list(xy),color=color,rx=rx,tx=tx,filtered=filtered,teardown=teardown)
        for i,xy in enumerate(self.input_path):
            tx=(['RAMP'] if xy in self.byxy else [])+([direction(xy,self.input_path[i+1])] if i+1<len(self.input_path) else [])
            add(xy,2,'RAMP' if i==0 else direction(xy,self.input_path[i-1]),tx,xy in self.byxy,True)
        def edge(parent,child,color):
            for rank in range(parent,child+1):
                xy=self.workers[rank]['xy']
                add(xy,color,'RAMP' if rank==child else direction(xy,self.workers[rank+1]['xy']),['RAMP' if rank==parent else direction(xy,self.workers[rank-1]['xy'])])
        for group in range(self.groups):
            for n in self.nodes[1:]:edge(group*40+n['parent'],group*40+n['rank'],3+2*(n['depth']-1)+n['side'])
            if group:edge((group-(group&-group))*40,group*40,self.gather(group)['color'])
        x,y=self.workers[0]['xy'];assert y==0
        for xx in range(x+1):add((xx,0),18,'RAMP' if xx==x else 'EAST',['RAMP' if xx==0 else 'WEST'])
        return list(routes.values())

    def document(self):
        routes=self.routes()
        return dict(schema='wse-complete-bf16-matrix-slice-v1',model=self.base['model'],revision=self.base['revision'],provenance=self.provenance,
                    application=[self.width,self.height],global_origin=self.origin,diagnostic_simprint=self.diagnostic,matrix=self.matrix,workers=self.workers,roots=[w for w in self.workers if w['k']==0],
                    input_path=[list(p) for p in self.input_path],routes=routes,gathers=[self.gather(g) for g in range(24)],
                    full_matrix_shape=[48,5120],full_model=False,gather_packet_words=2,gather_buffer_words_per_root=2,background_storage='Representative original mixed-bank prefixes at the exact candidate per-PE capacities; only target BF16 slot has complete frozen-atlas tensor identity.',
                    source_role='Temporary adjacent qualification source at global119,705; global resident role not admitted.',
                    scope='Complete original layer0 in_proj_a: all48 outputs and5120 inputs at frozen BF16 owner addresses. One2600-word broadcast,24 K40 trees, ordered concatenation and controller return. Host supplies independent BF16 input; not autonomous model inference.')

    def emit_layout(self):
        lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=631,.height=2});',
               'const filter=.{.kind=.{.counter=true},.count_data=true,.count_control=false,.init_counter=0,.limit1=2599,.max_counter=64};','layout {',' @set_rectangle(631,2);']
        active_input=set(self.input_path)
        for y in range(self.height):
            for x in range(self.width):
                w=self.byxy.get((x,y));n=self.nodes[w['k']] if w else self.nodes[0];children=n['children'] if w else [];g=self.gather(w['group']) if w and w['k']==0 else dict(steps=0,words=1,color=18)
                params=dict(role=0 if (x,y)==(0,0) else 1 if w else 2,input_on=(x,y) in active_input,fp8_slots=w['fp8_slots'] if w else 0,bf16_slots=w['bf16_slots'] if w else 0,k_rank=w['k'] if w else 0,children=len(children),subtree=n['size'] if w else 0,
                            left_color=3+2*(self.nodes[children[0]]['depth']-1)+self.nodes[children[0]]['side'] if children else 3,right_color=3+2*(self.nodes[children[1]]['depth']-1)+self.nodes[children[1]]['side'] if len(children)>1 else 4,
                            out_color=3+2*(n['depth']-1)+n['side'] if w and w['k'] else g['color'],gather_steps=g['steps'],gather_words=g['words'])
                params['trace']=self.diagnostic and ((x,y)==(0,0) or (w is not None and w['rank'] in [0,959]))
                text=','.join('.%s=%s'%(k,str(v).lower()) for k,v in params.items())
                text+=',.trace_tag='+json.dumps(('first' if w['rank']==0 else 'last') if w and params['trace'] else 'source' if params['trace'] else '')
                lines.append(f' @set_tile_code({x},{y},"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),{text}}});')
        for r in self.routes():
            x,y=r['xy'];extra=(', .teardown=true' if r['teardown'] else '')+(', .filter=filter' if r['filtered'] else '')
            lines.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}%s});'%(x,y,r['color'],r['rx'],','.join(r['tx']),extra))
        for name,dtype in [('bank','u32'),('target','u32'),('packet','u32'),('control','u16'),('local_result','f32'),('aggregate','u32'),('gather','u32'),('audit','u16'),('ticks','u16')]:lines.append(' @export_name("%s",[*]%s,%s);'%(name,dtype,str(name=='target').lower()))
        lines+=[' @export_name("initialize_aliases",fn()void);',' @export_name("arm",fn()void);',' @export_name("start",fn()void);',' @export_name("check_payload",fn(u16)void);','}']
        return '\n'.join(lines)+'\n'
