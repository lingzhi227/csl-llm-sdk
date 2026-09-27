"""Typed resident actors with arrival-triggered epochs and send-completion credits."""
from dataclasses import asdict,dataclass
from .mixed_bank import MixedBankPlan
from .contraction import tree

@dataclass(frozen=True)
class EpochBankPlan(MixedBankPlan):
 request_color:int=18
 result_color:int=17
 request_iq:int=4
 request_task:int=11
 request_dsr:int=2
 worker_uts:tuple=(2,3,4,5)
 controller_uts:tuple=(2,3)
 fixture_cases:int=12
 max_rounds:int=96
 @property
 def worker_height(self):return super().height
 @property
 def height(self):return self.worker_height+1
 def routes(self):
  routes=super().routes();h=self.worker_height
  def add(x,y,c,rx,tx):routes.append(dict(x=x,y=y,color=c,rx=rx,tx=tx))
  add(0,h,self.request_color,'RAMP',['NORTH'])
  for y in range(h):
   add(0,y,self.request_color,'SOUTH',['RAMP','EAST']+(['NORTH'] if y else []))
   add(1,y,self.request_color,'WEST',['RAMP'])
   add(0,y,self.result_color,'RAMP' if y==0 else 'NORTH',['SOUTH'])
  add(0,h,self.result_color,'NORTH',['RAMP'])
  return routes
 def validate(self):
  r=super().validate()
  if self.blocks!=(6,) or self.mesh_width!=2 or self.fixture_cases!=12 or self.max_rounds!=96:raise ValueError('Frozen representative epoch profile')
  used={self.color(v) for v in tree(6)[1:]}
  if len({self.request_color,self.result_color})!=2 or used&{self.request_color,self.result_color} or any(not 2<=v<=20 for v in [self.request_color,self.result_color]):raise ValueError('Event plane color collision')
  if self.request_iq in self.input_queues[:2] or not 2<=self.request_iq<=7 or self.request_task in self.tasks or not 8<=self.request_task<=20:raise ValueError('Request receive collision')
  if self.request_dsr!=2 or self.request_dsr in self.dsrs:raise ValueError('Request owns DSR2; memcpy owns DSR0')
  for ids,length in [(self.worker_uts,4),(self.controller_uts,2)]:
   if len(ids)!=length or len(set(ids))!=length or any(not 2<=v<=7 for v in ids):raise ValueError('Concurrent microthread collision')
  routes=self.routes();mapping={(v['x'],v['y'],v['color']):v for v in routes}
  if len(mapping)!=len(routes):raise ValueError('Duplicate route')
  steps={'NORTH':(0,-1,'SOUTH'),'SOUTH':(0,1,'NORTH'),'WEST':(-1,0,'EAST'),'EAST':(1,0,'WEST')}
  for v in routes:
   for direction in v['tx']:
    if direction=='RAMP':continue
    dx,dy,opposite=steps[direction];other=mapping.get((v['x']+dx,v['y']+dy,v['color']))
    if other is None or other['rx']!=opposite:raise ValueError('Disconnected link')
  r.update(application_pes=8,worker_pes=6,controller=[0,3],request_words=66,result_words=4,request_dsr=2,worker_microthreads=list(self.worker_uts),controller_microthreads=list(self.controller_uts),host_barrier_per_epoch=False,
   schedule='Request arrival resets local epoch, arms children and computes. Aggregate send callback alone grants next request receive. Controller advances only after its request send and result receive complete. Initial/final host audits are outside the autonomous loop.',
   timing='Same-controller request start through result arrival per epoch, plus full autonomous loop; last worker callbacks checked by final host audit. Preloaded independent activation fixture, not dependent neural/model tokens.')
  return r
 def document(self):return dict(schema='wse-resident-epochs-v1',plan=asdict(self),resources=self.validate(),trees=[tree(6)],routes=self.routes(),owners=[[self.coordinate(0,k) for k in range(6)]])
 def emit_layout(self):
  self.validate();nodes=tree(6);w,h=self.width,self.height
  lines=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={w},.height={h}}});','layout {',f' @set_rectangle({w},{h});']
  for v in nodes:
   rank=v['rank'];x,y=self.coordinate(0,rank);children=v['children']
   p=dict(rank=rank,children=len(children),subtree=v['size'],fp8_slots=self.fp8_slots,bf16_slots=self.bf16_slots,cases=self.fixture_cases,max_rounds=self.max_rounds,left_color=self.color(nodes[children[0]]) if children else 3,right_color=self.color(nodes[children[1]]) if len(children)>1 else 4,out_color=self.color(v) if rank else self.result_color,request_color=self.request_color,request_iq=self.request_iq,request_task=self.request_task,request_dsr=self.request_dsr)
   p.update(zip(('left_ut','right_ut','request_ut','send_ut'),self.worker_uts))
   p.update(zip(('left_iq','right_iq'),self.input_queues));p['tree_oq']=self.output_queues[0]
   p.update(zip(('left_task','right_task','send_task'),self.tasks));p.update(zip(('left_dsr','right_dsr','send_dsr'),self.dsrs))
   lines.append(' @set_tile_code(%d,%d,"worker.csl",.{.memcpy_params=memcpy.get_params(%d),%s});'%(x,y,x,','.join('.%s=%d'%(k,v) for k,v in p.items())))
  for x in range(w):lines.append(' @set_tile_code(%d,%d,"controller.csl",.{.memcpy_params=memcpy.get_params(%d),.active=%d,.cases=%d,.max_rounds=%d,.request_color=%d,.result_color=%d,.send_ut=%d,.recv_ut=%d});'%(x,h-1,x,int(x==0),self.fixture_cases,self.max_rounds,self.request_color,self.result_color,*self.controller_uts))
  for r in self.routes():lines.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});'%(r['x'],r['y'],r['color'],r['rx'],','.join(r['tx'])))
  for name,ty in [('weights','u16'),('scale','f32'),('bfweights','u16'),('request','u32'),('expected','f32'),('failure','u32'),('local_result','f32'),('aggregate','u32'),('audit','u16'),('requests','u32'),('results','u32'),('ticks','u16')]:lines.append(f' @export_name("{name}",[*]{ty},false);')
  lines+=[' @export_name("initialize",fn(u16,u16)void);',' @export_name("start",fn()void);','}'];return '\n'.join(lines)+'\n'
