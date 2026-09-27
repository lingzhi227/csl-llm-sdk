"""Static preorder binary contraction: two colors per level, two child queues.

Disjoint preorder intervals let each level/child side use non-overlapping express routes
without router switching. Full-K components remain distinct from full-M matrices.
"""
from dataclasses import asdict,dataclass

def tree(count):
 nodes=[None]*count
 def visit(first,size,depth,parent,side):
  child_sizes=[(size-1+1)//2,(size-1)//2] if size>1 else []
  children=[];offset=first+1
  for child_side,n in enumerate(child_sizes):
   if n:children.append(offset);visit(offset,n,depth+1,first,child_side);offset+=n
  nodes[first]=dict(rank=first,size=size,depth=depth,parent=parent,side=side,children=children)
 if count<1:raise ValueError('Empty contraction')
 visit(0,count,0,None,None);return nodes

@dataclass(frozen=True)
class ContractionPlan:
 blocks:tuple=(40,48,136)
 mesh_width:int=8
 first_tree_color:int=3
 chain_colors:tuple=(17,18)
 input_queues:tuple=(2,3,4)
 output_queues:tuple=(2,3)
 tasks:tuple=(8,9,10)
 dsrs:tuple=(3,5,6)
 @property
 def width(self):return self.mesh_width or max(self.blocks)
 @property
 def height(self):return sum(self.blocks)//self.mesh_width if self.mesh_width else len(self.blocks)
 def coordinate(self,group,rank):
  if not self.mesh_width:return (rank,group)
  row,col=divmod(rank,self.mesh_width);return (col if row%2==0 else self.mesh_width-1-col,sum(self.blocks[:group])//self.mesh_width+row)
 def color(self,node):return self.first_tree_color+2*(node['depth']-1)+node['side']
 def routes(self):
  routes={}
  def direction(origin,target):
   dx,dy=target[0]-origin[0],target[1]-origin[1]
   return {(1,0):'EAST',(-1,0):'WEST',(0,1):'SOUTH',(0,-1):'NORTH'}[dx,dy]
  def path(group,source,dest,color):
   for rank in range(dest,source+1):
    xy=self.coordinate(group,rank);x,y=xy
    r=dict(x=x,y=y,color=color,rx='RAMP' if rank==source else direction(xy,self.coordinate(group,rank+1)),tx=['RAMP' if rank==dest else direction(xy,self.coordinate(group,rank-1))])
    key=(x,y,color)
    if key in routes and routes[key]!=r:raise ValueError('Overlapping express routes')
    routes[key]=r
  for y,n in enumerate(self.blocks):
   for v in tree(n)[1:]:path(y,v['rank'],v['parent'],self.color(v))
   for x in range(1,n):path(y,x,x-1,self.chain_colors[x%2])
  return list(routes.values())
 def validate(self):
  if not self.blocks or any(n not in (40,48,136) for n in self.blocks) or len(self.blocks)>3:raise ValueError('Complete original K widths only')
  if self.mesh_width not in (0,4,8):raise ValueError('Supported line/4/8-wide geometry only')
  colors={self.color(v) for n in self.blocks for v in tree(n)[1:]}
  if colors & set(self.chain_colors) or len(set(self.chain_colors))!=2 or min(colors|set(self.chain_colors))<2 or max(colors|set(self.chain_colors))>20:raise ValueError('Reserved/overlapping colors')
  for ids,size,lo,hi in [(self.input_queues,3,2,7),(self.output_queues,2,2,7),(self.tasks,3,8,20),(self.dsrs,3,3,7)]:
   if len(ids)!=size or len(set(ids))!=size or any(not lo<=v<=hi for v in ids):raise ValueError('Resource collision')
  if 4 in self.dsrs:raise ValueError('Native dot owns DSR4')
  routes=self.routes();mapping={(r['x'],r['y'],r['color']):r for r in routes}
  owners=[self.coordinate(y,k) for y,n in enumerate(self.blocks) for k in range(n)]
  if len(set(owners))!=sum(self.blocks):raise ValueError('Owner collision')
  opposites={'EAST':'WEST','WEST':'EAST','NORTH':'SOUTH','SOUTH':'NORTH'}
  for y,n in enumerate(self.blocks):
   for v in tree(n)[1:]:
    for rank in range(v['rank'],v['parent'],-1):
     here=mapping[self.coordinate(y,rank)+(self.color(v),)];there=mapping[self.coordinate(y,rank-1)+(self.color(v),)]
     if len(here['tx'])!=1 or there['rx']!=opposites[here['tx'][0]]:raise ValueError('Disconnected tree link')
  return dict(active_pes=sum(self.blocks),application_pes=self.width*self.height,max_tree_edges=max(v['depth'] for n in self.blocks for v in tree(n)),tree_colors=len(colors),full_model=False,full_output_rows=False,stack_allowance=4096,sram_ceiling=48128,compiled_sram_admitted=False,
   schedule='Each node waits for own dot and both disjoint children, adds local then left then right, sends3words: two FP32 outputs and exact participant count. Separate right-to-left chain control.',
   timing='Per-row root start through complete full-K numerical result. Host arming/audit excluded and reported separately; no cross-PE clock subtraction.')
 def document(self):return dict(schema='wse-full-k-contraction-v1',plan=asdict(self),resources=self.validate(),trees=[tree(n) for n in self.blocks],routes=self.routes(),owners=[[self.coordinate(y,k) for k in range(n)] for y,n in enumerate(self.blocks)])
 def emit_layout(self):
  self.validate();w,h=self.width,self.height
  lines=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={w},.height={h}}});','layout {',f' @set_rectangle({w},{h});']
  for y,n in enumerate(self.blocks):
   nodes=tree(n)
   for rank in range(n if self.mesh_width else w):
    x,yy=self.coordinate(y,rank)
    v=nodes[rank] if rank<n else dict(children=[],parent=None);children=v['children']
    p=dict(rank=rank,count=n,active=int(rank<n),children=len(children),subtree=v.get('size',0),left_color=self.color(nodes[children[0]]) if children else 3,right_color=self.color(nodes[children[1]]) if len(children)>1 else 4,out_color=self.color(v) if 0<rank<n else 3,chain_in_color=self.chain_colors[(rank+1)%2],chain_out_color=self.chain_colors[rank%2])
    p.update(zip(('left_iq','right_iq','chain_iq'),self.input_queues));p.update(zip(('tree_oq','chain_oq'),self.output_queues));p.update(zip(('left_task','right_task','send_task'),self.tasks));p.update(zip(('left_dsr','right_dsr','send_dsr'),self.dsrs))
    lines.append(' @set_tile_code(%d,%d,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),%s});'%(x,yy,x,','.join('.%s=%d'%(k,v) for k,v in p.items())))
  for r in self.routes():lines.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});'%(r['x'],r['y'],r['color'],r['rx'],','.join(r['tx'])))
  for name,ty in [('weights','u16'),('scale','f32'),('raw','u32'),('packet','u32'),('local_result','f32'),('aggregate','u32'),('audit','u32'),('ticks','u16')]:lines.append(f' @export_name("{name}",[*]{ty},false);')
  lines+=[' @export_name("arm",fn(u16)void);',' @export_name("start",fn()void);','}'];return '\n'.join(lines)+'\n'
