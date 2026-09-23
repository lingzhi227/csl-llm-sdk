"""Source-only validation of a complete dense136 diagnostic boundary.

Inputs are role-qualified logical integer arrays reconstructed from future raw
captures. No SDK execution, host polling, or actual acceptance occurs here.
"""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact(values, expected, label):
    require(isinstance(values, list) and all(type(v) is int for v in values), label+' integer capture')
    require(values == expected, label+' mismatch')


def validate(snapshot, contract, ledger, serial, generation, position):
    require((serial,generation,position) in ((1,1,0),(2,1,1),(3,2,0)), 'Fixed serial identity')
    roles={(r['x'],r['y']):r for r in contract['roles']}
    expected_keys={f'{x},{y}' for x,y in roles}
    require(set(snapshot)==expected_keys,'Complete136-coordinate snapshot')
    ports={tuple(r['coordinate']):r for r in ledger['router_ports']}
    nodes={(n['x'],n['y']):n for n in contract['nodes']}
    for (x,y),role in roles.items():
        key=f'{x},{y}'; data=snapshot[key];kind=role['role']
        config_x=x-1 if role['root'] or kind in ('support','origin') else x
        exact(data['transport_config'],[config_x,y],key+' config')
        exact(data['controls'],[serial,generation,position,1 if serial==2 and (x,y)==(11,0) else 0],key+' controls')
        if kind=='router':
            n=nodes[x,y];p=ports[x,y]
            status=[0,n['mask'],x,y]+[serial*v for v in p['rx']]+[serial*v for v in p['tx']]
            status += [1 if n['mask'] & (1 << i) else 0 for i in range(3)]
            status += [3]*6+[serial,1 if serial==3 else 0]+[0]*11
            require(len(status)==32,'Router status schema')
            exact(data['transport_status'],status,key+' router status')
            continue
        support=kind in ('support','origin');root=role['root'];matrix=kind=='matrix'
        tx=3 if root else 2 if support else 0
        rx=1 if root else 6 if support else 0
        exact(data['transport_status'],[0,serial*tx,serial*tx,serial*tx,serial*rx,serial*rx,1,0],key+' packet status')
        state=[2*serial,serial,generation,position,2 if support else 0,2 if support else 0,
            0,0,0,2 if kind=='origin' else 0,0,0,1 if root else 0,1 if root else 0,0,0]
        exact(data['state'],state,key+' state')
        exact(data['frame_counters'],[2*serial,2*serial,768*serial,0] if kind=='origin' else [0]*4,key+' frame counters')
        if matrix:
            exact(data['lane_counters'],[serial,serial,serial,2*serial],key+' lane counters')
        if support:
            exact(data['receive_offsets'],[64,64],key+' receive offsets')
            exact(data['ack_pending'],[0,0],key+' ACK pending')
    # Verify every transported full128-BF16 row, independently of arithmetic.
    # Receiver slot0/1 maps to source row(destination_y+4-phase-1)%4.
    for y in range(4):
        rows=snapshot[f'11,{y}']['received_rows']
        require(isinstance(rows,list) and len(rows)==256,'Two disjoint complete receiver rows')
        require(all(type(v) is int and 0<=v<65536 for v in rows),'Receiver BF16 containers')
        for phase in range(2):
            source_y=(y+4-phase-1)%4
            source=snapshot[f'{1+5*phase},{source_y}']['rounded']
            exact(source,rows[phase*128:(phase+1)*128],f'phase{phase}/source{source_y} transport')
    return dict(scope='Complete captured boundary structure/identity/transport only; separate numeric oracle required',
        serial=serial,generation=generation,position=position,coordinates=136,full_rows=8,
        router_edges=30,all_boundary_checks_passed=True)


def validate_stable(first, second, contract, ledger, serial, generation, position):
    a=validate(first,contract,ledger,serial,generation,position)
    b=validate(second,contract,ledger,serial,generation,position)
    require(first==second,'Boundary changed across two complete snapshots')
    require(a==b,'Boundary validation identity')
    return a | dict(stable_snapshots=2)
