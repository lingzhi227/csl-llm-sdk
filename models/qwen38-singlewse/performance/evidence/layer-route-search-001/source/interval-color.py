import json,sys,collections,heapq
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import spatial.layer_routes as lr
from spatial.layer_routes import contraction_groups,reduction_routes
lr.edge_color=lambda node:node["route_color"]

def build(r):
 groups=list(contraction_groups(r))
 for g in groups:
  ends={}
  for a,b,rank in sorted((min(v['rank'],v['parent']),max(v['rank'],v['parent']),v['rank']) for v in g['nodes'] if v['parent'] is not None):
   c=next(c for c in range(3,18) if ends.get(c,-1)<a);ends[c]=b;g['nodes'][rank]['route_color']=c
 roots={tuple(g['root_pe']) for g in groups}
 # Match x-sorted roots to distinct nearby lower-payload actors, excluding roots.
 actors=sorted([a for a in r['quantization_actors'] if tuple(a['pe']) not in roots and a['pe'][1]==r['rect'][1]+r['rect'][3]-1],key=lambda a:tuple(a['pe']))
 order=sorted(groups,key=lambda g:tuple(g['root_pe']));n=len(order);m=len(actors)
 dp={(0,j):(0,[]) for j in range(m+1)}
 for i in range(1,n+1):
  for j in range(i,m+1):
   cost=sum(abs(x-y) for x,y in zip(order[i-1]['root_pe'],actors[j-1]['pe']))
   take=(dp[i-1,j-1][0]+cost,dp[i-1,j-1][1]+[j-1])
   dp[i,j]=min(take,dp.get((i,j-1),(10**9,[])))
 assigned={g['index']:actors[a] for g,a in zip(order,dp[n,m][1])}
 owner={q:next(g['index'] for g in groups if g['output_start']*8<=128*q<(g['output_start']+g['output_count'])*8) for q in range(136)}
 used={c:set() for c in [0,1,*range(3,18),19]}
 for g in groups:
  for flow in reduction_routes(r,g):
   for ent in flow['routes']:used[flow['color']].add(tuple(ent['pe']))
 flows=[]
 for g in groups:
  first=g['output_start']*8;end=first+g['output_count']*8
  for dest in sorted({owner[row//128] for row in range(first,end,8)}):
   a=assigned[dest]
   flows.extend([(g['index'],dest,'data',tuple(g['root_pe']),tuple(a['pe'])),(g['index'],dest,'credit',tuple(a['pe']),tuple(g['root_pe']))])
 endpoints={p for f in flows for p in f[3:]}
 x,y,w,h=r['rect']
 def route(src,dst,c):
  if src in used[c]:return None
  first=(*src,c);frontier=[(abs(src[0]-dst[0])+abs(src[1]-dst[1]),0,0,first)];parent={first:None};distance={first:(0,0)}
  while frontier:
   _,length,swaps,at=heapq.heappop(frontier)
   if distance[at]!=(length,swaps):continue
   if at[:2]==dst:
    p=[]
    while at is not None:p.append(at);at=parent[at]
    return p[::-1]
   ns=[(at[0]+1,at[1]),(at[0]-1,at[1]),(at[0],at[1]+1),(at[0],at[1]-1)]
   ns.sort(key=lambda z:abs(z[0]-dst[0])+abs(z[1]-dst[1]))
   for xy in ns:
    if not(x<=xy[0]<x+w and y<=xy[1]<y+h):continue
    for color in (at[2],at[2]^1):
     nxt=(*xy,color)
     cost=(length+1,swaps+(color!=at[2]))
     if color in used and xy not in used[color] and (xy==dst or xy not in endpoints) and cost<distance.get(nxt,(10**9,10**9)):
      parent[nxt]=at;distance[nxt]=cost;heapq.heappush(frontier,(cost[0]+abs(xy[0]-dst[0])+abs(xy[1]-dst[1]),*cost,nxt))
  return None
 result=[]
 for flow in sorted(flows,key=lambda f:-sum(abs(a-b) for a,b in zip(f[3],f[4]))):
  choices=[]
  for c in used:
   p=route(flow[3],flow[4],c)
   if p:choices.append((len(p),sum(a[2]!=b[2] for a,b in zip(p,p[1:])),c,p))
  if not choices:return {'failed':flow,'routed':len(result),'flows':len(flows)}
  *_,c,p=min(choices)
  for xx,yy,col in p:used[col].add((xx,yy))
  result.append((flow,c,p))
 return {'routed':len(result),'flows':len(flows),'max_hops':max(len(p)-1 for _,_,p in result),'route_entries':sum(len(p) for _,_,p in result),'colors':sorted(set(c for _,c,_ in result)),'assignments':[(g, a['pe']) for g,a in sorted(assigned.items())]}
plan=json.loads((Path(__file__).resolve().parents[1]/'evidence/layer-native-schedule-002/layer-schedule.json').read_text())
for s in [plan['stages'][1],plan['stages'][4]]:
 r=next(r for r in s['regions'] if r['role']=='gate_up');print(r['id'],json.dumps(build(r)))
