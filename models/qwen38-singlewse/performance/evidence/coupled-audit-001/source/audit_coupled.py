"""Independent complete-class forest and column-input coexistence audit."""
import hashlib,json,resource,time
from pathlib import Path
import numpy as np
from audit_projections import graph_audit,edges,require,digest

def audit(root):
 from spatial.forest import path_embedding,lower
 from spatial.columnar import lower_input_routes
 root=Path(root);started=time.monotonic()
 base=json.loads((root/'model-atlas.json').read_text());overlay=json.loads((root/'columnar-plan.json').read_text())
 require(digest(root/'model-atlas.json')==overlay['base_atlas_sha256'],'Original base identity')
 # Build the oracle's view directly rather than invoking candidate materialize.
 atlas=dict(base,classes=overlay['classes'],matrices=overlay['matrices'],columnar_overlay_sha256=digest(root/'columnar-plan.json'),columnar_base_atlas_sha256=digest(root/'model-atlas.json'))
 for name,h in overlay['preserved_base_fields_sha256'].items():
  require(hashlib.sha256(json.dumps(atlas[name],sort_keys=True,separators=(',',':')).encode()).hexdigest()==h,'Inherited field identity')
 plan=json.loads((root/'bundles.json').read_text());doc=json.loads((root/'forests.json').read_text());graph=json.loads((root/'configs/model-graph.json').read_text())
 require(plan['provenance']['atlas_sha256']==doc['atlas_sha256']==hashlib.sha256((json.dumps(atlas,indent=2)+'\n').encode()).hexdigest(),'Materialized atlas identity')
 require(plan['provenance']['overlay_sha256']==doc['columnar_overlay_sha256']==digest(root/'columnar-plan.json'),'Overlay identity')
 require(plan['provenance']['base_atlas_sha256']==overlay['base_atlas_sha256'],'Base identity')
 require(plan['provenance']['graph_sha256']==base['provenance']['model-graph.json']==digest(root/'configs/model-graph.json'),'Original graph identity')
 result=graph_audit(graph,atlas,plan)
 require(doc['ports']==dict(NONE=0,RAMP=1,NORTH=2,SOUTH=3,WEST=4,EAST=5) and doc['colors']==list(range(3,16)),'Port schema')
 expected_profiles=[('fp8',40,'fp8-main'),('fp8',48,'fp8-main'),('fp8',136,'fp8-wide'),('bf16',40,'bf16')]
 require([(p['kind'],p['k_blocks'],p['storage_class']) for p in doc['profiles']]==expected_profiles,'Profile/class coverage')
 rows=np.array([y for y in range(1160) if y not in base['geometry']['actor_rows']],np.int32)
 excluded=np.zeros((1160,750),bool)
 for rect in [g['state_rectangle'] for g in base['gdn']]+[base['state_class_spare_rectangle']]:
  x,y,w,h=rect;excluded[y:y+h,x:x+w]=True
 profiles=[];files={}
 for published,(kind,k,storage) in zip(doc['profiles'],expected_profiles):
  embedding=path_embedding(atlas,storage);xy,bankpos,_,_=embedding
  width,count,xoff=(748,857208,1) if storage=='fp8-wide' else (750,860880,0)
  ranks=np.arange(count);row,col=np.divmod(ranks,width)
  expected_xy=np.column_stack([xoff+np.where(row%2,width-1-col,col),rows[row]])
  require(np.array_equal(xy[bankpos],expected_xy),'Independent class coordinates')
  require(np.all((xy[:,0]>=0)&(xy[:,0]<750)&(xy[:,1]>=0)&(xy[:,1]<1160)),'Physical bounds')
  require(len(np.unique(xy[:,1]*750+xy[:,0]))==len(xy),'Path coordinate reuse')
  require(np.all(np.abs(np.diff(xy,axis=0)).sum(axis=1)==1),'Nonneighbor path step')
  bank=np.zeros(len(xy),bool);bank[bankpos]=True
  require(np.all(np.isin(xy[~bank,0],[xoff,xoff+width-1])) and np.all(np.isin(xy[~bank,1],base['geometry']['actor_rows'])),'Actor bridges')
  nodes=bankpos if kind=='fp8' else bankpos[~excluded[xy[bankpos,1],xy[bankpos,0]]]
  require(len(nodes)==atlas['classes'][storage]['pes'],'Complete class population')
  delta=np.diff(xy,axis=0);back=np.zeros(len(xy),np.uint16);forward=np.zeros_like(back)
  for x,y,f,r in [(1,0,5,4),(-1,0,4,5),(0,1,3,2),(0,-1,2,3)]:
   selected=(delta[:,0]==x)&(delta[:,1]==y);forward[:-1][selected]=f;back[1:][selected]=r
  words,actual_nodes=lower(atlas,kind,k,embedding);require(np.array_equal(actual_nodes,nodes),'Class participant identity')
  require(words.shape==(len(xy),13),'Complete dense forest size');groups=nodes.reshape(-1,k);relative=edges(k)
  compact=[dict(active=0,source=0,destination=0) for _ in range(k)];cuts=[0]*(k+1)
  for parent,child,plane in relative:
   bit=1<<plane;compact[parent]['destination']|=bit;compact[child]['source']|=bit
   for rank in range(parent,child+1):
    require(not compact[rank]['active']&bit,'Compact plane collision');compact[rank]['active']|=bit
   for rank in range(parent+1,child+1):cuts[rank]|=bit
  require(published['vertices']==compact and published['cuts']==cuts,'Published pattern differs from independent tree')
  nonnodes=np.ones(len(xy),bool);nonnodes[nodes]=False;active_total=holes=max_hops=0;active_colors=[]
  for plane in range(13):
   difference=np.zeros(len(xy)+1,np.int16);source=np.zeros(len(xy),bool);dest=np.zeros(len(xy),bool)
   for parent,child,color in relative:
    if color!=plane:continue
    first,last=groups[:,parent],groups[:,child];np.add.at(difference,first,1);np.add.at(difference,last+1,-1)
    source[last]=True;dest[first]=True;max_hops=max(max_hops,int(np.max(last-first)))
   occupancy=np.cumsum(difference[:-1]);require(np.all((occupancy==0)|(occupancy==1)),'PE/color collision')
   active=occupancy==1;expected=np.where(active,np.where(source,1,forward)|(np.where(dest,1,back)<<3),0)
   require(np.array_equal(words[:,plane],expected),'Independent complete forest mismatch')
   if active.any():active_colors.append(3+plane)
   active_total+=int(active.sum());holes+=int(np.count_nonzero(active&nonnodes))
  # Every complete dispatch maps each original row's K consecutive tiles onto
  # one of these physical trees, even across slot wraps and phase rotations.
  checked_tiles=0;root_count=0
  for b in plan['bundles']:
   if b['storage_class']!=storage or b['k_blocks']!=k:continue
   phase=atlas['classes'][storage]['phase'];pes=len(nodes)
   for seg in b['segments']:
    starts=(phase+seg['class_start']+np.arange(0,seg['count'],k,dtype=np.int64))%pes
    require(np.all(starts%k==0),'Dispatch splits actual tree')
    root_count+=len(starts);checked_tiles+=seg['count']
   for m in b['matrices']:
    original=atlas['matrices'][m['matrix']]
    require(original['tiles']==m['rows']//2*k,'Original matrix K conservation')
  inputs=None
  if kind=='fp8':
   ip=next(p for p in overlay['input_profiles'] if p['k_blocks']==k);incoming=lower_input_routes(base,ip)
   colors=[ip['horizontal_color']]+ip['vertical_colors']
   require(set(colors).isdisjoint(active_colors),'Input/reduction color collision')
   rank_at=np.full((1160,750),-1,np.int32);rank_at[xy[nodes,1],xy[nodes,0]]=np.arange(len(nodes))
   consumers=np.zeros((1160,750),np.uint8)
   for plane,color in enumerate(colors[1:],1):
    accepted=(incoming[plane]>>3)&1;consumers+=accepted.astype(np.uint8)
   require(np.array_equal(consumers,rank_at>=0),'Exact full input and contraction participant match')
   # Derive each needed block from the original physical class rank. Read the
   # published source/window descriptors without candidate source()/consumer().
   for record in ip['sources']:
    x=record['column'];lane=record['lane'];eligible=np.flatnonzero(consumers[:,x]&(((incoming[1+lane,:,x]>>3)&1)>0))
    expected_blocks=rank_at[eligible,x]%k
    require(np.all(np.isin(expected_blocks,record['blocks'])),'Input pair misses actual tree K block')
   inputs=dict(colors=colors,route_entries=int(incoming.size),exact_consumers=int(consumers.sum()),simultaneous_colors_disjoint=True)
   del incoming,rank_at,consumers
  name=kind+'-k'+str(k);file=root/(name+'.npy')
  with file.open('xb') as f:np.save(f,words,allow_pickle=False)
  files[file.name]=dict(bytes=file.stat().st_size,sha256=digest(file))
  profiles.append(dict(name=name,storage_class=storage,physical_path_pes=len(xy),actor_bridges=int((~bank).sum()),
   contributing_pes=len(nodes),independent_trees=len(groups),tree_edges=len(groups)*(k-1),route_entries=int(words.size),
   active_pe_color_entries=active_total,forwarding_hole_entries=holes,maximum_single_edge_physical_hops=max_hops,
   original_dispatch_tiles=checked_tiles,original_dispatch_row_pairs=root_count,input_coexistence=inputs))
 require(sum(p['original_dispatch_tiles'] for p in profiles)==105052160,'All original matrix tiles dispatched')
 return dict(schema='wse-coupled-input-forest-audit-v1',passed=True,physical=False,weights_read=False,graph=result,profiles=profiles,
  dense_forest_files=files,bundles_sha256=digest(root/'bundles.json'),forests_sha256=digest(root/'forests.json'),
  base_atlas_sha256=digest(root/'model-atlas.json'),overlay_sha256=digest(root/'columnar-plan.json'),
  duration_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  bf16_input_routes_admitted=False,output_scatter_admitted=False,route_epochs_admitted=False,compiled_sram_admitted=False,
  full_model_executable=False,full_model_speed_target_achieved=False,
  scope='All original graph/bundles bind new storage-class owners and complete forests. FP8 input endpoint and color coexistence checked across entire model geometry. No runtime queue/UT/SRAM, result return or physical execution admission.')
