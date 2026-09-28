"""Addressed command multicast and a single credited serpentine response bus."""
from spatial.layer_mlp_network import broadcast
from spatial.layer_schedule import rank_xy
from spatial.layer_routes import direction


def lower_device_network(rect,tags=None):
    x,y,w,h=rect;region=dict(rect=rect);count=w*h
    if count<2:raise ValueError('Gateway and internal endpoint required')
    pes=[rank_xy(region,i) for i in range(count)];gateway=pes[-1]
    # Existing broadcast lowering requires a bottom-row gateway. Serpentine
    # ordering provides that on either left or right depending on row parity.
    tags=list(range(count-1)) if tags is None else list(tags)
    if len(tags)!=count-1 or len(set(tags))!=len(tags) or any(type(t) is not int or not 0<=t<65535 for t in tags):
        raise ValueError('Missing/aliased internal device address')
    routes=broadcast(gateway,{tuple(pe):tag for pe,tag in zip(pes[:-1],tags)},12)
    workers=[]
    for i,pe in enumerate(pes):
        rx='RAMP' if i==0 else direction(pe,pes[i-1]);tx='RAMP' if i==count-1 else direction(pe,pes[i+1])
        routes.append(dict(pe=pe,color=13,rx=[rx],tx=[tx],exclusive_control_response=True))
        if i<count-1:workers.append(dict(pe=pe,tag=tags[i],reply_rx=rx,reply_tx=tx))
    result=dict(rect=rect,gateway=gateway,workers=workers,routes=routes,maximum_outstanding=1,
        command_encoding='Two tag16|payload16 words per logical32-bit word',
        response_rule='Only the addressed command owns injection. Restore transit RX after complete-response OQ flush.',
        compiled=False,executed=False,physical=False)
    result['audit']=audit_device_network(result);return result


def audit_device_network(plan):
    installed={};x,y,w,h=plan['rect'];delta={'NORTH':(0,-1),'SOUTH':(0,1),'WEST':(-1,0),'EAST':(1,0)}
    opposite={'NORTH':'SOUTH','SOUTH':'NORTH','WEST':'EAST','EAST':'WEST'}
    for r in plan['routes']:
        pe=r['pe'];key=(*pe,r['color'])
        if key in installed or r['color'] not in (12,13) or len(r['rx'])!=1 or not x<=pe[0]<x+w or not y<=pe[1]<y+h:
            raise ValueError('Control route ownership/boundary conflict')
        installed[key]=r
    workers={tuple(a['pe']):a for a in plan['workers']};gateway=tuple(plan['gateway'])
    if len(workers)!=w*h-1 or gateway in workers or len({a['tag'] for a in workers.values()})!=len(workers):raise ValueError('Control endpoint coverage')
    # Independently walk the multicast once, checking every tag's unique RAMP
    # predicate. No O(endpoint_count^2) walk is needed for3554 exact filters.
    todo=[(gateway,'RAMP')];seen=set();delivered={}
    while todo:
        pe,incoming=todo.pop();r=installed.get((*pe,12))
        if pe in seen or r is None or r['rx']!=[incoming]:raise ValueError('Cyclic/disconnected command multicast')
        seen.add(pe)
        for tx in r['tx']:
            if tx=='RAMP':
                if pe not in workers or r.get('filter_tag')!=workers[pe]['tag'] or 'filter_max' in r:raise ValueError('Wrong control address filter')
                if r['filter_tag'] in delivered:raise ValueError('Control command has multiple consumers')
                delivered[r['filter_tag']]=pe
            else:
                dx,dy=delta[tx];todo.append(((pe[0]+dx,pe[1]+dy),opposite[tx]))
    if set(seen)!={tuple(r['pe']) for r in plan['routes'] if r['color']==12} or set(delivered.values())!=set(workers):raise ValueError('Unreachable command endpoint')
    # Walk the reverse response line from the gateway. Its unique predecessor
    # must forward into this PE; every possible injection lies on that line.
    seen=set();pe=gateway
    while True:
        if pe in seen:raise ValueError('Response cycle')
        seen.add(pe);r=installed[(*pe,13)]
        if pe==gateway:
            if r['tx']!=['RAMP']:raise ValueError('Reply does not terminate at gateway')
        else:
            owner=workers[pe]
            if r['rx']!=[owner['reply_rx']] or r['tx']!=[owner['reply_tx']]:raise ValueError('Runtime reply directions differ from static bus')
        incoming=r['rx'][0]
        if incoming=='RAMP':break
        dx,dy=delta[incoming];previous=(pe[0]+dx,pe[1]+dy);link=installed.get((*previous,13))
        if link is None or link['tx']!=[opposite[incoming]]:raise ValueError('Disconnected credited response line')
        pe=previous
    if seen!=set(workers)|{gateway}:raise ValueError('Response path misses an internal PE')
    return dict(passed=True,endpoints=len(workers),command_color=12,response_color=13,maximum_response_hops=len(workers),physical=False)
