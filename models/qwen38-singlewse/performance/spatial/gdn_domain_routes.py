"""Lower explicit producer cohorts through multiple resident color domains.

The graph has a frontend domain and two child domains per group. Each child
bridge forwards one producer segment at a time. Return order is represented
explicitly; intermediate bridges advance the parent switch with a weighted
marker. Group order, cohort split, candidate width, palettes and SRAM reserves
are knobs. Static plans never imply compiler, neural or hardware acceptance.
"""
from collections import defaultdict, deque
from spatial.resident_gdn_routes import SWITCH_COLORS, direction


class Fabric:
    def __init__(self, profiles, census, routes, compact_reserve=1280, mixer_reserve=768):
        self.profiles = profiles
        self.cells = set(profiles)
        self.neighbors = {p: [q for q in ((p[0]-1,p[1]),(p[0]+1,p[1]),
                                          (p[0],p[1]-1),(p[0],p[1]+1)) if q in self.cells]
                          for p in self.cells}
        self.owners = {(*r['pe'],r['color']) for r in routes}
        if len(self.owners) != len(routes):
            raise ValueError('Existing color ownership is not unique')
        self.fronts = {p for p,c in profiles.items() if 'frontend_group' in c['parameters']}
        self.eligible = set()
        for p,c in profiles.items():
            a = c['parameters']
            compact = c['source']=='compact_layer_projection.csl' and not any(
                a.get(k,False) for k in ('fusion_actor','mlp_sender','output_sink'))
            mixer = c['source']=='device_mixer_layer_mlp_standby.csl' and not a['mixer_can_root']
            if (compact or mixer) and census[p] <= 48128-(compact_reserve if compact else mixer_reserve):
                self.eligible.add(p)
        self.census = census

    def free(self, pe, occupied):
        return {c for c in range(21) if (*pe,c) not in occupied}

    def common(self, cells, occupied):
        return set.intersection(*(self.free(pe, occupied) for pe in cells))

    def legal(self, color, occupied, front):
        return {pe for pe in self.cells if (*pe,color) not in occupied
                and (pe not in self.fronts or pe==front)}

    def flood(self, start, legal):
        if start not in legal:
            return {}
        parents = {start:None}
        queue = [start]
        for here in queue:
            for pe in self.neighbors[here]:
                if pe in legal and pe not in parents:
                    parents[pe]=here
                    queue.append(pe)
        return parents

    def tree(self, start, targets, legal):
        parents=self.flood(start,legal)
        if not targets <= parents.keys():
            return None
        used={start:None}
        for pe in sorted(targets):
            while pe not in used:
                used[pe]=parents[pe]
                pe=parents[pe]
        return used

    def chain(self, start, anchors, legal, nearest=False):
        if not {start,*anchors}<=legal:
            return None
        left=set(anchors);rank={pe:i for i,pe in enumerate(anchors)};path=[start]
        while left:
            parents={path[-1]:None};queue=deque([path[-1]]);used=set(path[:-1]);found=[]
            preferred=min(left,key=rank.__getitem__)
            while queue:
                here=queue.popleft()
                for pe in self.neighbors[here]:
                    if pe not in legal or pe in parents or pe in used:
                        continue
                    parents[pe]=here
                    if pe in left:
                        found.append(pe)
                        if nearest or pe==preferred:
                            queue.clear()
                            break
                    else:
                        queue.append(pe)
            if not found:
                return None
            pe=found[0] if nearest else min(found,key=rank.__getitem__)
            target=pe;segment=[]
            while pe is not None:
                segment.append(pe);pe=parents[pe]
            path.extend(reversed(segment[:-1]));left.remove(target)
        return path


def orders(cells):
    for axis in (0,1):
        for reverse in (False,True):
            result=[]
            for i,k in enumerate(sorted({p[axis]for p in cells},reverse=reverse)):
                result.extend(sorted((p for p in cells if p[axis]==k),
                                     key=lambda p:p[1-axis],reverse=bool(i%2)))
            yield result
            yield list(reversed(result))


def emit_domain(source, targets, forward, returning, tree, path, workers, weighted=False):
    """Emit directly inspectable PE/color records; counters affect worker ingress."""
    children=defaultdict(list)
    for pe,parent in tree.items():
        if parent is not None:
            children[parent].append(pe)
    rows=[]
    for pe,parent in sorted(tree.items()):
        row=dict(pe=list(pe),color=forward,rx=['RAMP' if pe==source else direction(pe,parent)],
                 tx=(['RAMP']if pe in targets else[])+[direction(pe,q)for q in children[pe]])
        if pe in targets and not weighted:
            row['counter']=dict(initial=(1161-387*(workers[pe]['head']%3))%1161,limit=1160,maximum=386)
        if not row['tx']:
            raise ValueError('Unconsumed broadcast leaf')
        rows.append(row)
    for i,pe in enumerate(path):
        parent=direction(pe,path[i-1])if i else'RAMP'
        child=direction(pe,path[i+1])if i+1<len(path)else None
        row=dict(pe=list(pe),color=returning,rx=['RAMP']if pe in targets else[child],tx=[parent])
        if pe in targets and (not weighted or child is not None):
            row['switch']=dict(next_rx=child or parent,pop_on_advance=True)
        if None in row['rx']:
            raise ValueError('Unterminated return chain')
        rows.append(row)
    if len({(*r['pe'],r['color'])for r in rows})!=len(rows):
        raise ValueError('Domain owns a PE/color twice')
    return rows


def split_cohorts(group, profiles, workers):
    front=next(pe for pe,c in profiles.items()if c['parameters'].get('frontend_group')==group)
    selected={tuple(w['pe']):w for w in workers if w['head']//3==group}
    left={pe for pe,w in selected.items()if w['role']=='gate_up'and pe[0]<=front[0]}
    right=set(selected)-left
    if not left or not right:
        raise ValueError('Explicit two-cohort split requires both cohorts')
    return front,selected,(left,right)


def candidates(fabric, front, cohort, workers, occupied, taken, width=12):
    """Bounded pool of legal child domains, sorted by actual route geometry."""
    colors=fabric.common(cohort,occupied)
    components={}
    for color in colors:
        legal=fabric.legal(color,occupied,front)
        component=set(fabric.flood(min(cohort),legal))
        if cohort<=component:
            components[color]=component
    result=[]
    pairs=[(a,b)for a in components for b in components if a!=b and b in SWITCH_COLORS]
    # A small local connected component is useful, not a globally free color.
    pairs.sort(key=lambda pair:(len(components[pair[0]])+len(components[pair[1]]),pair))
    for forward,returning in pairs:
        possible=(fabric.eligible-taken-set(workers))&components[forward]&components[returning]
        possible={pe for pe in possible if len(fabric.common({front,pe},occupied)-{forward,returning})>=2}
        possible=sorted(possible,key=lambda p:(min(abs(p[0]-q[0])+abs(p[1]-q[1])for q in cohort),
                                                abs(p[0]-front[0])+abs(p[1]-front[1]),p))
        for bridge in possible[:24]:
            paths=[]
            nearest=fabric.chain(bridge,sorted(cohort),components[returning],nearest=True)
            if nearest:
                paths.append(nearest)
            for order in orders(cohort):
                path=fabric.chain(bridge,order,components[returning])
                if path:
                    paths.append(path)
            if not paths:
                continue
            path=min(paths,key=lambda p:(len(p),p))
            tree=fabric.tree(bridge,cohort,components[forward])
            rows=emit_domain(bridge,cohort,forward,returning,tree,path,workers)
            sequence=[p for p in path if p in cohort]
            result.append(dict(source=bridge,colors=(forward,returning),tree=tree,path=path,routes=rows,
                               keys={(*r['pe'],r['color'])for r in rows},workers=sequence,
                               segments=[workers[p]['columns']//2 for p in sequence],
                               cost=len(tree)+len(path)-2))
        # Retain color diversity, avoiding a pool filled by one local pair.
        if len(result)>=width*4:
            break
    result.sort(key=lambda c:(c['cost'],c['source'],c['colors']))
    chosen=[];counts=defaultdict(int)
    for candidate in result:
        if counts[candidate['colors']]>=max(2,width//4):
            continue
        counts[candidate['colors']]+=1;chosen.append(candidate)
        if len(chosen)>=width:
            break
    return chosen


def viable_future(fabric, definitions, occupied, taken):
    """Necessary connected-domain and available-cohost guard, not a route proof."""
    for group,(front,workers,cohorts)in definitions.items():
        for index,cohort in enumerate(cohorts):
            common=fabric.common(cohort,occupied);components={}
            for color in common:
                component=set(fabric.flood(min(cohort),fabric.legal(color,occupied,front)))
                if cohort<=component:
                    components[color]=component
            good=False
            for forward,a in components.items():
                for returning,b in components.items():
                    if forward==returning or returning not in SWITCH_COLORS:
                        continue
                    bridges=(a&b&fabric.eligible)-taken-set(workers)
                    if any(len(fabric.common({front,p},occupied)-{forward,returning})>=2 for p in bridges):
                        good=True;break
                if good:
                    break
            if not good:
                return dict(group=group,cohort=index,reason='No connected two-color child domain with an eligible bridge')
    return None


def lower(profiles, workers, census, retained, groups, candidate_width=12, on_group=None):
    fabric=Fabric(profiles,census,retained)
    definitions={g:split_cohorts(g,profiles,workers)for g in groups}
    if len(definitions)!=len(groups):
        raise ValueError('Duplicate group')
    occupied=set(fabric.owners);taken=set();plans=[];all_routes=[];rejections=[]
    for group in groups:
        front,ws,cohorts=definitions[group]
        pools=[candidates(fabric,front,cohort,ws,occupied,taken,candidate_width)for cohort in cohorts]
        found=None
        for a in pools[0]:
            for b in pools[1]:
                if a['source']==b['source']or a['keys']&b['keys']:
                    continue
                bridges={a['source'],b['source']};child_keys=a['keys']|b['keys']
                current=occupied|child_keys
                colors=fabric.common(bridges|{front},current)-set(a['colors'])-set(b['colors'])
                for forward in sorted(colors):
                    tree=fabric.tree(front,bridges,fabric.legal(forward,current,front))
                    if tree is None:
                        continue
                    for returning in sorted((colors&SWITCH_COLORS)-{forward}):
                        legal=fabric.legal(returning,current,front)
                        options=[fabric.chain(front,order,legal)for order in [sorted(bridges),sorted(bridges,reverse=True)]]
                        options=[p for p in options if p]
                        if not options:
                            continue
                        path=min(options,key=lambda p:(len(p),p))
                        root_rows=emit_domain(front,bridges,forward,returning,tree,path,ws,weighted=True)
                        root_keys={(*r['pe'],r['color'])for r in root_rows}
                        if root_keys&current:
                            raise ValueError('Root overwrote an existing owner')
                        future={g:definitions[g]for g in groups if g!=group and g not in {p['group']for p in plans}}
                        reason=viable_future(fabric,future,current|root_keys,taken|bridges)
                        if reason:
                            rejections.append(dict(candidate_group=group,**reason));continue
                        ordered=[pe for pe in path if pe in bridges]
                        children=[]
                        for child in (a,b):
                            children.append(dict(bridge=list(child['source']),colors=list(child['colors']),
                                workers=[list(pe)for pe in child['workers']],segments=child['segments'],
                                frames=sum(child['segments']),markers=len(child['segments']),
                                switch_parent=child['source']!=ordered[-1],
                                tree=[dict(pe=list(p),parent=list(q)if q else None)for p,q in sorted(child['tree'].items())],
                                path=[list(p)for p in child['path']],measured_base_bytes=census[child['source']]))
                        rows=a['routes']+b['routes']+root_rows
                        found=dict(group=group,frontend=list(front),root_colors=[forward,returning],
                                   root_tree=[dict(pe=list(p),parent=list(q)if q else None)for p,q in sorted(tree.items())],
                                   root_path=[list(p)for p in path],children=children,routes=rows)
                        break
                    if found:break
                if found:break
            if found:break
        if not found:
            return dict(passed=False,unresolved_group=group,candidate_pool_sizes=list(map(len,pools)),
                        plans=plans,routes=all_routes,guard_rejections=rejections)
        keys={(*r['pe'],r['color'])for r in found['routes']}
        if keys&occupied or len(keys)!=len(found['routes']):
            raise ValueError('Accepted group overwrites route ownership')
        occupied|=keys;taken|={tuple(c['bridge'])for c in found['children']}
        plans.append(found);all_routes.extend(found['routes'])
        if on_group:
            on_group(found)
    return dict(passed=True,plans=plans,routes=all_routes,guard_rejections=rejections)
