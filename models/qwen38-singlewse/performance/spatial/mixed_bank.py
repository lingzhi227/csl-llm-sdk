"""Resident mixed precision bank plus static tree, explicitly partial-K qualification."""
from dataclasses import asdict,dataclass
from .contraction import ContractionPlan,tree

@dataclass(frozen=True)
class MixedBankPlan(ContractionPlan):
 blocks:tuple=(6,)
 mesh_width:int=2
 fp8_slots:int=112
 bf16_slots:int=12
 def routes(self):
  # Reuse P9's checked express-route construction, omit its chain control.
  return [r for r in super().routes() if r['color'] not in self.chain_colors]
 def validate(self):
  if len(self.blocks)!=1 or not 2<=self.blocks[0]<=16 or self.mesh_width not in (1,2,4) or self.blocks[0]%self.mesh_width:raise ValueError('Representative bank geometry')
  if (self.fp8_slots,self.bf16_slots)!=(112,12):raise ValueError('Capacity cannot be reduced for qualification')
  colors={self.color(v) for v in tree(self.blocks[0])[1:]}
  if colors & set(self.chain_colors) or min(colors)<2 or max(colors)>20:raise ValueError('Reserved colors')
  for ids,size,lo,hi in [(self.input_queues,3,2,7),(self.output_queues,2,2,7),(self.tasks,3,8,20),(self.dsrs,3,3,7)]:
   if len(ids)!=size or len(set(ids))!=size or any(not lo<=v<=hi for v in ids):raise ValueError('Resource collision')
  if {4,7}&set(self.dsrs):raise ValueError('Dot owns DSR4 and DSR7')
  owners=[self.coordinate(0,k) for k in range(self.blocks[0])]
  if len(set(owners))!=self.blocks[0]:raise ValueError('Owner collision')
  self.routes()
  return dict(application_pes=self.width*self.height,active_pes=self.blocks[0],full_model=False,full_input_width=False,full_output_rows=False,stack_allowance=4096,sram_ceiling=48128,compiled_sram_admitted=False,
   resident_payload_bytes_per_pe=112*260+12*512,control_bytes_per_pe=4,per_tile_descriptor_bytes=0,
   descriptor_leases=dict(child_receives=list(self.dsrs[:2]),send=self.dsrs[2],native_dot=4,bf16_expansion=7),
   metadata_policy='Checked [kind,slot] u16 control per PE; caller supplies ownership. No per-tile metadata array. Complete model address schedule not implemented.',
   schedule='Arm all child receives, host readiness audit, local typed resident dot then local/left/right ordered tree. No chain comparison.',
   timing='Root own start to result, same-clock only; host control/operand upload and arming excluded and separately reported.')
 def document(self):return dict(schema='wse-mixed-bank-component-v1',plan=asdict(self),resources=self.validate(),trees=[tree(self.blocks[0])],routes=self.routes(),owners=[[self.coordinate(0,k) for k in range(self.blocks[0])]])
 def emit_layout(self):
  self.validate();nodes=tree(self.blocks[0]);w,h=self.width,self.height
  lines=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={w},.height={h}}});','layout {',f' @set_rectangle({w},{h});']
  for v in nodes:
   rank=v['rank'];x,y=self.coordinate(0,rank);children=v['children']
   p=dict(rank=rank,children=len(children),subtree=v['size'],fp8_slots=self.fp8_slots,bf16_slots=self.bf16_slots,left_color=self.color(nodes[children[0]]) if children else 3,right_color=self.color(nodes[children[1]]) if len(children)>1 else 4,out_color=self.color(v) if rank else 3)
   p.update(zip(('left_iq','right_iq'),self.input_queues));p['tree_oq']=self.output_queues[0]
   p.update(zip(('left_task','right_task','send_task'),self.tasks));p.update(zip(('left_dsr','right_dsr','send_dsr'),self.dsrs))
   lines.append(' @set_tile_code(%d,%d,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),%s});'%(x,y,x,','.join('.%s=%d'%(k,v) for k,v in p.items())))
  for r in self.routes():lines.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});'%(r['x'],r['y'],r['color'],r['rx'],','.join(r['tx'])))
  for name,ty in [('weights','u16'),('scale','f32'),('bfweights','u16'),('control','u16'),('packet','u32'),('local_result','f32'),('aggregate','u32'),('audit','u32'),('ticks','u16')]:lines.append(f' @export_name("{name}",[*]{ty},false);')
  lines+=[' @export_name("arm",fn()void);',' @export_name("start",fn()void);','}'];return '\n'.join(lines)+'\n'
