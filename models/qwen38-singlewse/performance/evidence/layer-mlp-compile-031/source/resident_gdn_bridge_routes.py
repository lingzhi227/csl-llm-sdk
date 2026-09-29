"""Separate multicast and ordered-return lowering across explicit color domains.

One translator joins two domains. Every retained PE/color owner is immutable.
An unsolved group fails explicitly; callers cannot label a partial route plan a
complete neural graph. Cohort membership, group ordering, palettes and bridge
resource allowance are manual lowering controls, not opaque layout heuristics.
"""
from collections import defaultdict
from spatial.resident_gdn_routes import SWITCH_COLORS,direction


def lower(profiles,workers,census,retained,existing,groups=(15,14),source_columns=range(10,15),bridge_reserve=4000):
    cells=set(profiles);busy={(*r['pe'],r['color'])for r in retained+existing}
    if len(busy)!=len(retained)+len(existing):raise ValueError('Existing route ownership collides')
    if len(set(groups))!=len(groups)or not set(groups)<=set(range(3,16)):raise ValueError('New group selection')
    palette=set(range(21));taken=set();protected={};plans=[]
    def free(pe):return {c for c in palette if(*pe,c)not in busy}
    def common(ps):return set.intersection(*(free(pe)for pe in ps))
    def partition(g):
        ws={tuple(w['pe']):w for w in workers if w['head']//3==g}
        front=next(pe for pe,c in profiles.items()if c['parameters'].get('frontend_group')==g)
        source={pe for pe,w in ws.items()if w['role']=='mix'or(g in source_columns and w['role']=='gate_up'and pe[0]<=front[0])}
        return ws,front,source,set(ws)-source
    # Reserve every future cohort's common endpoint choices. Transit on other
    # colors remains legal; this is more precise than forbidding whole PEs.
    for g in range(3,16):
        ws,f,up,down=partition(g)
        for endpoints in (up|{f},down):
            for color in common(endpoints):
                for pe in endpoints:protected[(*pe,color)]=g
        for color in palette:protected[(*f,color)]=g
    eligible=[]
    for pe,c in profiles.items():
        a=c['parameters']
        plain=c['source']=='device_mixer_layer_mlp_standby.csl'and not a['mixer_can_root']
        plain|=c['source']=='compact_layer_projection.csl'and not any(a.get(k,False)for k in ('fusion_actor','mlp_sender','output_sink'))
        if plain and census[pe]<=48128-bridge_reserve:eligible.append(pe)
    def neighbors(pe):
        x,y=pe;return ((x-1,y),(x+1,y),(x,y-1),(x,y+1))
    def priorities(ps):
        for axis in (0,1):
            for reverse in (False,True):
                order=[]
                for i,k in enumerate(sorted({p[axis]for p in ps},reverse=reverse)):
                    order+=sorted([p for p in ps if p[axis]==k],key=lambda p:p[1-axis],reverse=bool(i%2))
                yield order;yield order[::-1]
    for group in groups:
        ws,front,up,down=partition(group)
        def allowed(pe,color):return (*pe,color)not in busy and protected.get((*pe,color),group)==group
        def chain(start,anchors,color):
            legal={pe for pe in cells if allowed(pe,color)}
            if not {start,*anchors}<=legal:return None
            path=[start];left=set(anchors);rank={pe:i for i,pe in enumerate(anchors)}
            while left:
                parents={path[-1]:None};queue=[path[-1]];found=[];used=set(path[:-1])
                for here in queue:
                    for pe in neighbors(here):
                        if pe not in legal or pe in parents or pe in used:continue
                        parents[pe]=here
                        if pe in left:found.append(pe)
                        else:queue.append(pe)
                if not found:return None
                target=min(found,key=lambda pe:rank[pe]);segment=[];cursor=target
                while cursor is not None:segment.append(cursor);cursor=parents[cursor]
                path+=segment[::-1][1:];left.remove(target)
            return path
        def tree(start,targets,color):
            legal={pe for pe in cells if allowed(pe,color)}
            if start not in legal:return None
            parents={start:None};queue=[start]
            for here in queue:
                for pe in neighbors(here):
                    if pe not in legal or pe in parents:continue
                    parents[pe]=here;queue.append(pe)
            if not targets<=parents.keys():return None
            used={start:None}
            for target in sorted(targets):
                while target not in used:used[target]=parents[target];target=parents[target]
            return used
        colors=common(down);found=None
        for forward in sorted(colors):
            for reverse in sorted(colors&SWITCH_COLORS):
                if forward==reverse:continue
                pair=(forward,reverse)
                bridges=[pe for pe in eligible if pe not in taken and pe not in ws and set(pair)<=free(pe)]
                bridges.sort(key=lambda p:(min(abs(p[0]-q[0])+abs(p[1]-q[1])for q in down)+abs(p[0]-front[0])+abs(p[1]-front[1]),p))
                for bridge in bridges:
                    first_colors=common({front,bridge,*up})-set(pair)
                    first_pairs=[(a,b)for a in sorted(first_colors)for b in sorted(first_colors&(SWITCH_COLORS if up else palette))if a!=b]
                    if not first_pairs:continue
                    for order in priorities(down):
                        reverse_path=chain(bridge,order,reverse)
                        if reverse_path:break
                    if not reverse_path:continue
                    for incoming,returning in first_pairs:
                        up_path=chain(front,sorted(up,reverse=True)+[bridge],returning)
                        if not up_path or up_path[-1]!=bridge:continue
                        up_tree=tree(front,up|{bridge},incoming);down_tree=tree(bridge,down,forward)
                        if up_tree is None or down_tree is None:continue
                        found=dict(group=group,frontend=list(front),bridge=list(bridge),incoming=[incoming,returning],outgoing=list(pair),
                            source_workers=[list(p)for p in sorted(up)],child_workers=[list(p)for p in sorted(down)],
                            child_frames=sum(ws[p]['columns']//2 for p in down),child_markers=len(down),
                            up_path=[list(p)for p in up_path],down_path=[list(p)for p in reverse_path],
                            up_tree=[dict(pe=list(p),parent=list(q)if q else None)for p,q in sorted(up_tree.items())],
                            down_tree=[dict(pe=list(p),parent=list(q)if q else None)for p,q in sorted(down_tree.items())],
                            bridge_measured_base_bytes=census[bridge],bridge_unmeasured_reserve=bridge_reserve)
                        break
                    if found:break
                if found:break
            if found:break
        if not found:raise ValueError(f'Group{group} cannot be lowered through one admissible bridge; retain it as unrouted or split another domain')
        for key,color in [('up_path',found['incoming'][1]),('down_path',found['outgoing'][1])]:
            for pe in found[key]:
                if (*pe,color)in busy:raise ValueError('Ordered return route collision')
                busy.add((*pe,color))
        for key,color in [('up_tree',found['incoming'][0]),('down_tree',found['outgoing'][0])]:
            for row in found[key]:
                if (*row['pe'],color)in busy:raise ValueError('Broadcast route collision')
                busy.add((*row['pe'],color))
        taken.add(tuple(found['bridge']));plans.append(found)
    routes=[];by_pe={tuple(w['pe']):w for w in workers}
    for plan in plans:
        for domain in ('up','down'):
            source=tuple(plan['frontend']if domain=='up'else plan['bridge'])
            selected={tuple(p)for p in plan['source_workers'if domain=='up'else'child_workers']}
            incoming,returning=plan['incoming'if domain=='up'else'outgoing']
            tree_rows=plan[domain+'_tree'];children=defaultdict(list)
            for row in tree_rows:
                if row['parent']is not None:children[tuple(row['parent'])].append(tuple(row['pe']))
            for row in tree_rows:
                pe=tuple(row['pe']);ramp=pe in selected or(domain=='up'and pe==tuple(plan['bridge']))
                r=dict(pe=list(pe),color=incoming,rx=['RAMP'if pe==source else direction(pe,tuple(row['parent']))],
                       tx=(['RAMP']if ramp else[])+[direction(pe,q)for q in children[pe]])
                if pe in selected:r['counter']=dict(initial=(1161-387*(by_pe[pe]['head']%3))%1161,limit=1160,maximum=386)
                if not r['tx']:raise ValueError('Unconsumed multicast leaf')
                routes.append(r)
            path=[tuple(p)for p in plan[domain+'_path']]
            for i,pe in enumerate(path):
                parent=direction(pe,path[i-1])if i else'RAMP';child=direction(pe,path[i+1])if i+1<len(path)else None
                is_bridge=domain=='up'and pe==tuple(plan['bridge'])
                r=dict(pe=list(pe),color=returning,rx=['RAMP']if pe in selected or is_bridge else[child],tx=[parent])
                if pe in selected:r['switch']=dict(next_rx=child or parent,pop_on_advance=True)
                if None in r['rx']:raise ValueError('Unterminated return route')
                routes.append(r)
    if len({(*r['pe'],r['color'])for r in routes})!=len(routes):raise ValueError('Emitted route collision')
    return dict(schema='resident-gdn-bridged-domains-v1',groups=list(groups),plans=plans,routes=routes,
                original_routes_unchanged=True,original_banks_unchanged=True,new_software_relays=len(plans),
                all_groups_routed=False,compiled=False,executed=False,physical=False,full_layer=False,
                scope='Separate broadcast trees and ordered return chains with explicit weighted-terminal bridges; not numerical or complete-layer qualification.')
