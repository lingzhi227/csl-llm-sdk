"""Complete model ownership with compact bank rings and distributed actor rows.

This is an ownership/lifetime lowering, not compiled full-model route or SRAM admission.
No weight values or hundred-million-entry tile tables are materialized here.
"""
from bisect import bisect_left,bisect_right
from collections import Counter
from dataclasses import asdict,dataclass
import hashlib,json,math,re
from pathlib import Path

@dataclass(frozen=True)
class AtlasPolicy:
 width:int=750
 height:int=1160
 actor_first_row:int=48
 actor_row_stride:int=96
 actor_row_count:int=12
 bank_pes:int=860880
 gdn_key_shards:int=4
 gdn_value_shards:int=4
 attention_token_shards:int=8
 attention_channel_shards:int=4
 context:int=96
 canonical_page_bytes:int=32768
 quant_workers:int=4352
 quant_items:int=4
 quant_group_workers:int=32
 capture_bytes_per_quant_pe:int=28672
 sram_ceiling:int=48128
 def validate(self):
  if (self.width,self.height,self.bank_pes)!=(750,1160,860880):raise ValueError('Qualified geometry domain')
  if self.actor_row_count!=12 or self.actor_row_stride<9 or self.actor_first_row<0 or self.actor_rows[-1]+8>=self.height:raise ValueError('Actor/state bands overlap or leave fabric')
  if self.bank_pes%math.lcm(40,48,136):raise ValueError('FP8 ring breaks K groups')
  if (self.gdn_key_shards,self.gdn_value_shards,self.attention_token_shards,self.attention_channel_shards,self.context)!=(4,4,8,4,96):raise ValueError('Complete pinned short-context state partition')
  if (self.quant_workers,self.quant_items,self.quant_group_workers)!=(4352,4,32):raise ValueError('Complete group128 and17408-value producer domain')
  if self.canonical_page_bytes!=32768 or self.capture_bytes_per_quant_pe!=28672 or self.sram_ceiling!=48128:raise ValueError('Storage ceilings cannot be relaxed')
 @property
 def actor_rows(self):return tuple(self.actor_first_row+i*self.actor_row_stride for i in range(self.actor_row_count))

class Geometry:
 def __init__(self,policy):
  policy.validate();self.policy=policy;self.rows=tuple(y for y in range(policy.height) if y not in policy.actor_rows)
  if len(self.rows)*policy.width<policy.bank_pes:raise ValueError('Insufficient bank coordinates')
 def bank_xy(self,index):
  p=self.policy
  if not 0<=index<p.bank_pes:raise ValueError('Bank index')
  r,c=divmod(index,p.width);return (c if r%2==0 else p.width-1-c,self.rows[r])
 def bank_id(self,x,y):
  p=self.policy
  if not (0<=x<p.width and 0<=y<p.height):raise ValueError('Coordinate bounds')
  r=bisect_left(self.rows,y)
  if r==len(self.rows) or self.rows[r]!=y:return None
  b=r*p.width+(x if r%2==0 else p.width-1-x)
  return b if b<p.bank_pes else None
 def actors(self):
  p=self.policy;out=[(x,y) for y in p.actor_rows for x in range(p.width)]
  for b in range(p.bank_pes,len(self.rows)*p.width):
   r,c=divmod(b,p.width);out.append((c if r%2==0 else p.width-1-c,self.rows[r]))
  return sorted(out,key=lambda xy:(xy[1],xy[0]))

def complement_segments(total,excluded):
 """Sorted interval bijection between compact eligible IDs and physical bank IDs."""
 excluded=sorted(excluded)
 if len(set(excluded))!=len(excluded) or any(not 0<=v<total for v in excluded):raise ValueError('Invalid exclusion')
 segments=[];start=0;compact=0
 for value in excluded+[total]:
  if value>start:
   segments.append(dict(bank_start=start,compact_start=compact,count=value-start));compact+=value-start
  start=value+1
 return segments,compact

def eligible_to_bank(index,segments):
 if index<0:raise ValueError('Negative eligible index')
 i=bisect_right([s['compact_start'] for s in segments],index)-1
 if i<0 or index>=segments[i]['compact_start']+segments[i]['count']:raise ValueError('Eligible index')
 return segments[i]['bank_start']+index-segments[i]['compact_start']

def bank_to_eligible(index,segments):
 i=bisect_right([s['bank_start'] for s in segments],index)-1
 if i<0 or index>=segments[i]['bank_start']+segments[i]['count']:return None
 return segments[i]['compact_start']+index-segments[i]['bank_start']

def tile_address(matrix,tile,classes,segments,geometry):
 if not 0<=tile<matrix['tiles']:raise ValueError('Matrix tile domain')
 c=classes[matrix['kind']];absolute=matrix['stream_start']+tile;owner=(c['phase']+absolute)%c['pes'];slot=absolute//c['pes']
 bank=owner if matrix['kind']=='fp8' else eligible_to_bank(owner,segments)
 return dict(bank=bank,xy=geometry.bank_xy(bank),slot=slot,row_tile=tile//matrix['k_blocks'],k_block=tile%matrix['k_blocks'])

def slot_count(span,owner,pes,phase):return span//pes+int((owner-phase)%pes<span%pes)

def intervals_for(matrix,classes):
 """One or more complete-K ring segments; each descriptor owns at most one tile/PE."""
 c=classes[matrix['kind']];n=c['pes'];absolute=matrix['stream_start'];remaining=matrix['tiles'];relative=0;out=[]
 while remaining:
  residue=absolute%n;size=min(remaining,n-residue)
  assert residue%matrix['k_blocks']==size%matrix['k_blocks']==0
  out.append(dict(class_start=residue,count=size,slot=absolute//n,tile_start=relative,row_start=relative//matrix['k_blocks']))
  absolute+=size;relative+=size;remaining-=size
 return out

def value_arena(graph,policy):
 """Linear scan with closed operation lifetimes,32-cell alignment and coalescing.

One16-byte cell holds four logical values (BF16 uses its low8 bytes). Cells stripe
across the quant/value actor pool, preserving128-element group boundaries.
"""
 producers={};last={}
 for node in graph['nodes']:
  for v in node['outputs']:producers[v]=node['id'];last[v]=node['id']
  for v in node['inputs']:
   if v!='token':last[v]=node['id']
 last[graph['selected_token']]=len(graph['nodes'])
 free=[];active={};records={};high=0;peak_live=0
 def release(base,size):
  free.append((base,size));free.sort();merged=[]
  for b,n in free:
   if merged and merged[-1][0]+merged[-1][1]==b:merged[-1]=(merged[-1][0],merged[-1][1]+n)
   else:merged.append((b,n))
  free[:]=merged
 for node in graph['nodes']:
  for name in list(active):
   if last[name]<node['id']:
    v=active.pop(name);release(v['base_cell'],v['reserved_cells'])
  for name in node['outputs']:
   spec=graph['values'][name];elements=math.prod(spec['shape']);cells=math.ceil(elements/4);reserved=math.ceil(cells/32)*32;base=None
   for j,(b,n) in enumerate(free):
    if n>=reserved:
     base=b;free.pop(j)
     if n>reserved:free.insert(j,(b+reserved,n-reserved))
     break
   if base is None:base=high;high+=reserved
   rec=dict(shape=spec['shape'],dtype=spec['dtype'],elements=elements,base_cell=base,cells=cells,reserved_cells=reserved,birth=node['id'],last_use=last[name])
   records[name]=rec;active[name]=rec
  peak_live=max(peak_live,sum(v['reserved_cells'] for v in active.values()))
 return dict(values=records,cell_bytes=16,items_per_cell=4,alignment_cells=32,high_water_cells=high,peak_live_reserved_cells=peak_live,
  local_cells_per_actor=math.ceil(high/policy.quant_workers),bytes_per_actor=16*math.ceil(high/policy.quant_workers),
  owner_formula='quant_actor_id=(value.base_cell+element_index//4)%4352',slot_formula='local_cell=(value.base_cell+element_index//4)//4352; lane=element_index%4',
  lifetime='Producer through last consumer inclusive; input/output storage never aliases during an operation; selected token retained through feedback.')

def build(root,policy=AtlasPolicy()):
 root=Path(root);g=Geometry(policy);p=policy
 paths={n:root/'configs'/n for n in ['model-graph.json','tensors.json','config.json','rotary-frequencies.json']}
 graph,header,config,rotary=[json.loads(paths[n].read_text()) for n in paths]
 config=config.get('text_config',config)
 if graph['model']!='Qwen/Qwen3.8-27B-FP8' or graph['revision']!='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a' or header['revision']!=graph['revision'] or graph['context']!=p.context:raise ValueError('Pinned original graph required')
 for key,value in dict(hidden_size=5120,num_hidden_layers=64,linear_num_value_heads=48,linear_num_key_heads=16,linear_key_head_dim=128,linear_value_head_dim=128,num_key_value_heads=4,num_attention_heads=24,head_dim=256).items():
  if config[key]!=value:raise ValueError('Pinned model dimensions')
 specs=header['tensors'];names=list(dict.fromkeys(w for node in graph['nodes'] for w in node['weights']))
 if len(names)!=1251 or len(graph['nodes'])!=1172:raise ValueError('Incomplete graph')
 gdn_layers=sorted(int(n['attributes']['state'].split('.')[1]) for n in graph['nodes'] if n['op']=='gated_delta_recurrence')
 attn_layers=sorted(int(n['attributes']['state'].split('.')[1]) for n in graph['nodes'] if n['op']=='kv_append')
 if len(gdn_layers)!=48 or len(attn_layers)!=16 or set(gdn_layers+attn_layers)!=set(range(64)):raise ValueError('Full layer/state coverage')
 actors=g.actors();actor_index={xy:i for i,xy in enumerate(actors)};used={};state_banks=set();gdn=[];attention=[];anchors={}
 def claim(xy,role):
  xy=tuple(xy)
  if xy not in actor_index or xy in used:raise ValueError('Actor overlap/non-actor coordinate: '+str(xy))
  used[xy]=role;return actor_index[xy]
 for i,layer in enumerate(gdn_layers):
  stripe=i//4;lane=(i%4)//2;depth=i%2;ay=p.actor_rows[stripe];base=lane*192;anchors[layer]=(base+depth,ay)
  for head in range(48):
   x=base+4*head;y=ay+1+depth*4;ctrl=(x+depth,ay);cid=claim(ctrl,'gdn_controller')
   for yy in range(y,y+4):
    for xx in range(x,x+4):
     b=g.bank_id(xx,yy)
     if b is None or b in state_banks:raise ValueError('State/actor overlap')
     state_banks.add(b)
   gdn.append(dict(layer=layer,value_head=head,key_head=head//3,controller=cid,state_rectangle=[x,y,4,4],key_shards=4,value_shards=4,state_shape_per_pe=[32,32],state_bytes_per_pe=4096,
    conv_channel_starts=[(head//3)*128,2048+(head//3)*128,4096+head*128],history_bytes=3072,parameter_bytes=3332))
 semantic_state_banks=set(state_banks)
 # One explicit spare4x4 block makes the BF16 eligible ring divisible by40.
 sx,sy=384,p.actor_rows[0]+1
 for y in range(sy,sy+4):
  for x in range(sx,sx+4):
   b=g.bank_id(x,y)
   if b is None or b in state_banks:raise ValueError('Spare state-class overlap')
   state_banks.add(b)
 segments,bf_pes=complement_segments(p.bank_pes,state_banks)
 if bf_pes%40 or len(state_banks)!=36880 or len(semantic_state_banks)!=36864:raise ValueError('State/ring conservation')
 #16 attention layers over12 rows: the final four rows each host two layers.
 row_attention_count=Counter()
 for i,layer in enumerate(attn_layers):
  stripe=i if i<12 else i-4;ordinal=row_attention_count[stripe];row_attention_count[stripe]+=1;ay=p.actor_rows[stripe];base=384+ordinal*132;anchors[layer]=(base,ay)
  for head in range(4):
   x=base+head*33;controller=claim((x,ay),'attention_controller');shards=[]
   for ts in range(8):
    for ds in range(4):shards.append(dict(actor=claim((x+1+ts*4+ds,ay),'attention_kv'),token_start=ts*12,tokens=12,channel_start=ds*64,channels=64,kv_bytes=3072))
   attention.append(dict(layer=layer,kv_head=head,query_heads=list(range(head*6,head*6+6)),controller=controller,shards=shards,parameter_bytes=1152))
 def allocate_near(anchor,role):
  remaining=(xy for xy in actors if xy not in used)
  xy=min(remaining,key=lambda q:(abs(q[0]-anchor[0])+abs(q[1]-anchor[1]),q[1],q[0]));return claim(xy,role)
 # Canonical non-matrix tensors are packed by layer in32768-byte pages.
 nonmat=[n for n in names if len(specs[n]['shape'])!=2];buckets={};canonical={};pages=[]
 for name in nonmat:
  match=re.search(r'\.layers\.(\d+)\.',name);key=int(match[1]) if match else 64;buckets.setdefault(key,[]).append(name)
 for layer,nn in sorted(buckets.items()):
  total=sum(specs[n]['bytes'] for n in nn);anchor=anchors.get(layer,(p.width//2,p.actor_rows[-1]));ids=[allocate_near(anchor,'canonical_weights') for _ in range(math.ceil(total/p.canonical_page_bytes))]
  pages.extend(dict(actor=a,layer=layer,capacity_bytes=p.canonical_page_bytes) for a in ids);offset=0
  for name in nn:
   if specs[name]['dtype']!='BF16':raise ValueError('Non-matrix storage dtype')
   size=specs[name]['bytes'];chunks=[];consumed=0
   while consumed<size:
    page,local=divmod(offset,p.canonical_page_bytes);take=min(size-consumed,p.canonical_page_bytes-local)
    chunks.append(dict(actor=ids[page],actor_byte_offset=local,tensor_byte_offset=consumed,bytes=take));offset+=take;consumed+=take
   canonical[name]=dict(shape=specs[name]['shape'],dtype='BF16',bytes=size,chunks=chunks)
  assert offset==total
 global_controller=allocate_near((p.width//2,p.height//2),'global_control')
 remaining=[xy for xy in actors if xy not in used]
 if len(remaining)<p.quant_workers:raise ValueError('No complete parallel producer pool')
 quant_ids=[claim(xy,'quant_value') for xy in remaining[:p.quant_workers]]
 arena=value_arena(graph,p)
 norms=[]
 for node in graph['nodes']:
  if node['op']=='zero_centered_rmsnorm':
   name=node['weights'][0];value=node['inputs'][0];v=arena['values'][value]
   if specs[name]['shape']!=[5120] or v['elements']!=5120:raise ValueError('Norm input/gain shape')
   norms.append(dict(node=node['id'],weight=name,input=value,bank=len(norms),owner_start=v['base_cell']%p.quant_workers,owners=1280,elements_per_owner=4))
 if len(norms)!=129:raise ValueError('All original hidden normalizations required')
 classes={'fp8':dict(pes=p.bank_pes,phase=0,tile_bytes=260,stream_span=0,real_tiles=0,padding_tiles=0),
          'bf16':dict(pes=bf_pes,phase=bf_pes//2,tile_bytes=512,stream_span=0,real_tiles=0,padding_tiles=0)}
 if classes['bf16']['phase']%40:raise ValueError('Phase breaks BF16 K groups')
 matrices=[];scales={};bound=set(nonmat)
 scale_names={n+'_scale_inv' for n in names if specs[n]['dtype']=='F8_E4M3'}
 for name in names:
  if name in scale_names or name in canonical:continue
  spec=specs[name];m,k=spec['shape'];kind='fp8' if spec['dtype']=='F8_E4M3' else 'bf16';c=classes[kind];blocks=k//128
  if spec['dtype'] not in ('F8_E4M3','BF16') or m%2 or k%128 or blocks not in (40,48,136):raise ValueError('Unsupported original tile shape')
  start=math.ceil(c['stream_span']/blocks)*blocks;tiles=(m//2)*blocks;pad=start-c['stream_span'];c['stream_span']=start+tiles;c['real_tiles']+=tiles;c['padding_tiles']+=pad
  record=dict(id=len(matrices),tensor=name,kind=kind,shape=[m,k],row_tiles=m//2,k_blocks=blocks,tiles=tiles,stream_start=start,padding_before=pad,mode='lookup' if name=='model.language_model.embed_tokens.weight' else 'gemv')
  record['segments']=intervals_for(record,classes);matrices.append(record);bound.add(name)
  if kind=='fp8':
   sn=name+'_scale_inv'
   if specs[sn]['shape']!=[math.ceil(m/128),blocks] or specs[sn]['dtype']!='BF16':raise ValueError('Original scale identity')
   scales[sn]=dict(matrix=record['id'],source_index='[output_row//128,k_block]',resident='one exact FP32 expansion per2x128 tile');bound.add(sn)
 if bound!=set(names) or len(matrices)!=498:raise ValueError('Lost original tensors')
 # Complete PE census is small metadata; no original values are loaded.
 histogram=Counter();payload=0
 for bank in range(p.bank_pes):
  nf=slot_count(classes['fp8']['stream_span'],bank,p.bank_pes,0);bid=bank_to_eligible(bank,segments)
  nb=0 if bid is None else slot_count(classes['bf16']['stream_span'],bid,bf_pes,classes['bf16']['phase'])
  state=4096 if bank in state_banks else 0;histogram[nf,nb,state]+=1;payload+=nf*260+nb*512
  if nf>111 or nb>13 or nf*260+nb*512+state>=p.sram_ceiling:raise ValueError('Bank data overflow')
 expected_payload=sum(c['stream_span']*c['tile_bytes'] for c in classes.values());assert payload==expected_payload
 capture_elements=p.context*(64*5120+5120+248320);capture_bytes=capture_elements*2
 if capture_bytes>p.capture_bytes_per_quant_pe*p.quant_workers:raise ValueError('Full diagnostic capture reservation')
 profiles=[dict(fp8_slots=f,bf16_slots=b,state_capacity_bytes=state,pes=count,data_bytes=f*260+b*512+state,remaining_code_stack_scratch_bytes=p.sram_ceiling-(f*260+b*512+state)) for (f,b,state),count in sorted(histogram.items())]
 roles=[dict(id=i,xy=xy,role=used.get(xy,'spare')) for i,xy in enumerate(actors)]
 original_bytes=sum(specs[n]['bytes'] for n in names)
 if original_bytes!=29468003328:raise ValueError('Original payload identity')
 return dict(schema='wse-complete-model-atlas-v1',model=graph['model'],revision=graph['revision'],policy=asdict(p),context=p.context,
  geometry=dict(bank_rows=list(g.rows),actor_rows=list(p.actor_rows),actor_tail_pe_count=len(actors)-p.actor_row_count*p.width,bank_order='Serpentine over non-actor rows; truncate final120 logical positions'),
  classes=classes,bf16_eligible_segments=segments,state_class_spare_rectangle=[sx,sy,4,4],matrices=matrices,scale_aliases=scales,canonical_tensors=canonical,canonical_pages=pages,
  gdn=gdn,attention=attention,actors=roles,global_controller=global_controller,quant_actor_ids=quant_ids,value_arena=arena,normalizations=norms,
  quant_storage=dict(transient_bytes_per_pe=arena['bytes_per_actor'],normalization_gain_capacity_bytes_per_pe=129*4*2,capture_capacity_bytes_per_pe=p.capture_bytes_per_quant_pe,
   remaining_code_stack_other_scratch_bytes=p.sram_ceiling-arena['bytes_per_actor']-129*4*2-p.capture_bytes_per_quant_pe,capture_required_bytes=capture_bytes,capture_contents='BF16 all64 layer outputs, final norm and complete vocabulary for every96 positions; optional validation mode, not a speed claim'),
  bank_profiles=profiles,metrics=dict(original_tensors=len(names),matrices=len(matrices),original_payload_bytes=original_bytes,resident_matrix_bytes_including_padding_scales=payload,canonical_nonmatrix_bytes=sum(specs[n]['bytes'] for n in nonmat),
   bank_pes=p.bank_pes,actor_pes=len(actors),actor_roles=dict(Counter(x['role'] for x in roles)),gdn_state_pes=len(semantic_state_banks),gdn_state_fp32_bytes=len(semantic_state_banks)*4096,spare_state_capacity_pes=16,
   replicated_gdn_history_bytes=len(gdn)*3072,replicated_gdn_parameter_bytes=len(gdn)*3332,kv_cache_bytes=sum(s['kv_bytes'] for a in attention for s in a['shards']),
   gdn_state_elements_per_pe=1024,attention_elements_per_k_or_v_shard=12*64,canonical_pages=len(pages),segment_descriptors=sum(len(m['segments']) for m in matrices)),
  provenance={n:hashlib.sha256(path.read_bytes()).hexdigest() for n,path in paths.items()},
  physical_routes_admitted=False,compiled_sram_admitted=False,full_model_executable=False,full_model_speed_target_achieved=False,
  unresolved=['Compile co-resident FP8/state and variable mixed-bank profiles with actual code/stack/scratch','Lower matrix-specific multicast and tagged reductions over eligible rings','Implement distributed GDN/attention state arithmetic with frozen numerical qualification','Compose quant/value/normalization/capture roles under actual SRAM','Generate complete event schedule and dependent token feedback; measure physical full model'])
