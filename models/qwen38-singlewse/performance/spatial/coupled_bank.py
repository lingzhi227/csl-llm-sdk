"""Bounded composed bank engine: input windows, native dots, tree, root return."""
from .filtered_bank import FilteredBankPlan
from .contraction import tree

class CoupledBankPlan(FilteredBankPlan):
    def coordinate(self,rank):
        y,x=divmod(rank,3);return (1+x if y%2==0 else 3-x,y+1)
    def nodes(self):
        nodes=tree(12)
        return [dict(n,xy=self.coordinate(n['rank']),color=3+2*(n['depth']-1)+n['side'] if n['rank'] else 16) for n in nodes]
    def document(self):
        d=super().document();d['schema']='wse-coupled-bank-component-v1';d['tree']=self.nodes();d['return_color']=16;d['decoder_qualification']=dict(packed_patterns=65536,passive_pe_ranges=[[0,y,(y-1)*16384,16384] for y in range(1,5)])
        d['scope']='Two-stage counter windows -> twelve original maximum mixed banks -> ordered twelve-participant tree -> source-controller return. Independent host-provided operands; not a complete original K width/matrix or autonomous model feedback.'
        d['leases']=dict(input=dict(iq=2,ut=2,dsr=2),left=dict(iq=3,ut=3,dsr=3),right=dict(iq=4,ut=4,dsr=5),send=dict(oq=2,ut=5,dsr=6),native_dsrs=[4,7],controller_return=dict(iq=3,ut=4,dsr=3))
        return d
    def emit_layout(self):
        base=super().emit_layout().splitlines();nodes=self.nodes();byxy={n['xy']:n for n in nodes};out=[]
        for line in base:
            if '@set_tile_code' in line:
                import re
                x,y=map(int,re.search(r'@set_tile_code\((\d+),(\d+)',line).groups());n=byxy.get((x,y));children=n['children'] if n else []
                values=dict(test_first=(y-1)*16384 if x==0 and y>0 else 0,rank=n['rank'] if n else 0,children=len(children),subtree=n['size'] if n else 0,
                            left_color=nodes[children[0]]['color'] if children else 3,right_color=nodes[children[1]]['color'] if len(children)>1 else 4,out_color=n['color'] if n else 16)
                line=line[:-3]+','+','.join('.%s=%d'%(k,v) for k,v in values.items())+'});'
            if '@export_name("audit"' in line:line=line.replace('[*]u32','[*]u16')
            if '@export_name("arm"' in line:
                for child in nodes[1:]:
                    for rank in range(child['parent'],child['rank']+1):
                        xy=self.coordinate(rank)
                        def direction(other):
                            b=self.coordinate(other);return {(1,0):'EAST',(-1,0):'WEST',(0,1):'SOUTH',(0,-1):'NORTH'}[(b[0]-xy[0],b[1]-xy[1])]
                        rx='RAMP' if rank==child['rank'] else direction(rank+1);tx='RAMP' if rank==child['parent'] else direction(rank-1)
                        out.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});'%(xy[0],xy[1],child['color'],rx,tx))
                for x,y,rx,tx in [(1,1,'RAMP','WEST'),(0,1,'EAST','NORTH'),(0,0,'SOUTH','RAMP')]:
                    out.append(' @set_color_config(%d,%d,@get_color(16),.{.routes=.{.rx=.{%s},.tx=.{%s}}});'%(x,y,rx,tx))
                out.append(' @export_name("test_unpack",fn()void);')
                out.append(' @export_name("probe",[*]u32,false);')
                out.append(' @export_name("aggregate",[*]u32,false);')
                out.append(' @export_name("left",[*]u32,false);')
                out.append(' @export_name("right",[*]u32,false);')
            out.append(line)
        return '\n'.join(out)+'\n'
