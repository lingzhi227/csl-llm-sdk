"""First production GDN chains on unchanged original bank coordinates.

Groups0..2 have compatible static color domains. Other groups require explicit
translation or joint placement; this lowerer fails closed instead of displacing
retained routes or treating those groups as connected. No software relay is
inserted, and no arithmetic or bank address is changed.
"""
from collections import deque

COLORS = {0: (14, 9), 1: (15, 1), 2: (19, 0)}
SWITCH_COLORS = {0,1,2,3,4,5,6,7,8,9,12,13,16,17,20,21}


def direction(a, b):
    delta = (b[0]-a[0], b[1]-a[1])
    if delta not in {(1,0),(-1,0),(0,1),(0,-1)}:
        raise ValueError('Nonadjacent physical hop')
    return {(1,0):'EAST',(-1,0):'WEST',(0,1):'SOUTH',(0,-1):'NORTH'}[delta]


def path_between(start, end, legal, blocked):
    parents = {start: None}; queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == end:
            path = []
            while current is not None:
                path.append(current); current = parents[current]
            return list(reversed(path))
        x,y = current
        neighbors = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
        neighbors.sort(key=lambda pe:(abs(pe[0]-end[0])+abs(pe[1]-end[1]),pe))
        for pe in neighbors:
            if pe in legal and pe not in parents and (pe not in blocked or pe == end):
                parents[pe] = current; queue.append(pe)
    raise ValueError(f'No collision-free chain segment from {start} to {end}')


def lower(stage, workers, frontends, retained, groups=(0,1,2)):
    groups = tuple(groups)
    if not groups or len(set(groups)) != len(groups) or set(groups)-COLORS.keys():
        raise ValueError('Only explicitly supported production groups0..2')
    ox,oy,width,height = stage['rect']
    cells = {(x,y) for x in range(ox,ox+width) for y in range(oy,oy+height)}
    occupied = {(*r['pe'],r['color']) for r in retained}
    if len(occupied) != len(retained):
        raise ValueError('Retained route collisions')
    routes = []; chains = []
    for group in groups:
        incoming, returning = COLORS[group]
        if returning not in SWITCH_COLORS:
            raise ValueError('Return color has no hardware switch')
        selected = {tuple(w['pe']):w for w in workers if w['head']//3 == group}
        front = tuple(frontends[group])
        # Preserve actual P45 ownership. Its first three upper cohorts are full
        # rectangles; validate this before emitting their serpentine traversal.
        upper = {pe for pe,w in selected.items() if w['role'] == 'gate_up'}
        xs = sorted({pe[0] for pe in upper}, reverse=True)
        ys = sorted({pe[1] for pe in upper}, reverse=True)
        if not upper or upper != {(x,y) for x in xs for y in ys}:
            raise ValueError('Selected upper cohort is no longer rectangular')
        outside = sorted(selected.keys()-upper, reverse=True)
        anchors = outside + [(x,y) for i,x in enumerate(xs) for y in (ys if i%2==0 else ys[::-1])]
        legal = {pe for pe in cells if all((*pe,c) not in occupied for c in (incoming,returning))}
        if not {front,*selected}.issubset(legal):
            raise ValueError('Selected endpoint collides with retained routing')
        path = [front]
        for index, target in enumerate(anchors):
            blocked = set(path[:-1]) | set(anchors[index+1:])
            segment = path_between(path[-1], target, legal, blocked)
            path.extend(segment[1:])
        if len(set(path)) != len(path) or set(path)&selected.keys() != selected.keys():
            raise ValueError('Incomplete or cyclic production chain')
        for index,pe in enumerate(path):
            parent = direction(pe,path[index-1]) if index else None
            child = direction(pe,path[index+1]) if index+1<len(path) else None
            forward = dict(pe=list(pe),color=incoming,rx=[parent or 'RAMP'],
                           tx=(['RAMP'] if pe in selected else [])+([child] if child else []))
            reverse = dict(pe=list(pe),color=returning,
                           rx=['RAMP'] if pe in selected else [child],tx=[parent or 'RAMP'])
            if pe in selected:
                head = selected[pe]['head']%3
                forward['counter'] = dict(initial=(1161-387*head)%1161,limit=1160,maximum=386)
                reverse['switch'] = dict(next_rx=child or parent,pop_on_advance=True)
            if not forward['tx'] or None in reverse['rx']:
                raise ValueError('Unterminated production path')
            for r in (forward,reverse):
                key = (*pe,r['color'])
                if key in occupied:
                    raise ValueError('Lowered route displaces an existing owner')
                occupied.add(key); routes.append(r)
        chains.append(dict(group=group,frontend=list(front),input_color=incoming,return_color=returning,
                           path=[list(pe) for pe in path],workers=len(selected),
                           state_elements=sum(128*w['columns'] for w in selected.values()),
                           return_frames=sum(w['columns']//2 for w in selected.values()),
                           return_markers=len(selected),hops=len(path)-1,
                           worker_order=[list(pe) for pe in path if pe in selected]))
    return dict(schema='resident-gdn-static-chains-v1',rect=stage['rect'],groups=list(groups),
                chains=chains,routes=routes,retained_route_entries=len(retained),
                new_route_entries=len(routes),original_banks_unchanged=True,
                new_software_relays=0,all_groups_routed=set(groups)==set(range(16)),
                compiled=False,executed=False,physical=False,full_layer=False,
                scope='Static original-coordinate connectivity only; endpoint ELF admission and '
                      'numerical execution are separate. Other groups are deliberately unrouted.')


def emit(plan):
    ox,oy = plan['rect'][:2]; lines = []
    for r in plan['routes']:
        fields = '.routes=.{.rx=.{%s},.tx=.{%s}}' % (','.join(r['rx']),','.join(r['tx']))
        if 'counter' in r:
            c = r['counter']
            fields += ',.filter=.{.kind=.{.counter=true},.count_data=true,.init_counter=%d,.limit1=%d,.max_counter=%d}' % (c['initial'],c['limit'],c['maximum'])
        if 'switch' in r:
            fields += ',.switches=.{.pos1=.{.rx=%s},.pop_mode=.{.pop_on_advance=true}}' % r['switch']['next_rx']
        lines.append(' @set_color_config(%d,%d,@get_color(%d),.{%s});' % (r['pe'][0]-ox,r['pe'][1]-oy,r['color'],fields))
    return '\n'.join(lines)+'\n'
