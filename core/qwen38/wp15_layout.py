"""Single source of WP15 arrays, copies, launches and planned footprint."""
from .wp14_layout import PRODUCER_ARRAYS
from .qk_rope_protocol import ARRAYS as QK, INITIAL as QK_INIT, PROBES
from .attention_protocol import ARRAYS as ATTN, INITIAL as ATTN_INIT
from .projected_attention_protocol import FRAGMENTS,fabric_budget,routes

COMMON={'request_control':(4,32),'request_state':(8,32),'handoff_control':(10,32),
        'handoff_state':(32,32),'wire_data':(142,32),'wire_ack':(15,32)}
QK_ARRAYS={'qk_'+n:v for n,v in QK.items()}
ATTENTION_ARRAYS={'attention_'+n:v for n,v in ATTN.items() if n!='control'}
ARRAYS={**PRODUCER_ARRAYS,**QK_ARRAYS,**ATTENTION_ARRAYS,**COMMON}
QK_INITIAL=('qk_input',*('qk_'+n for n in QK_INIT))
ATTENTION_INITIAL=('attention_input',*('attention_'+n for n in ATTN_INIT))
QK_READS=tuple(QK_ARRAYS)
ATTENTION_READS=tuple(ATTENTION_ARRAYS)
COMMANDS=('initialize_request','begin','accumulate','finalize','arm_handoff','handoff','preprocess','attend','release')


def integer(name):
    return ARRAYS[name][1]==16 or name in (*COMMON,'control','state','weight_guard_snapshot') or name.endswith(('_state','_events','_control'))


def sequence():
    rows=[]
    def cp(direction,name,pe):
        count,bits=ARRAYS[name]
        rows.append(dict(kind='copy',direction=direction,name=name,pe=pe,count=count,bits=bits,
                         host_bytes=count*4,native_bytes=count*bits//8))
    def launch(name):rows.append(dict(kind='launch',name=name))
    for n in QK_INITIAL:cp('h2d',n,8)
    for n in ATTENTION_INITIAL:cp('h2d',n,9)
    for pe in range(10):cp('h2d','request_control',pe)
    launch('initialize_request')
    for pe in range(10):cp('d2h','request_state',pe)
    for n in ('attention_state','attention_cache'):cp('d2h',n,9)
    for pe in range(8):
        for n in ('output_storage','bf16_storage'):cp('h2d',n,pe)
    for call in range(2):
        for pe in range(10):cp('h2d','request_control',pe)
        cp('h2d','qk_control',8)
        for pe in range(8):cp('h2d','control',pe)
        launch('begin')
        for pe in (8,9):cp('d2h','handoff_state',pe)
        cp('d2h','attention_cache',9)
        for tile in range(46):
            for pe in range(8):
                for n in ('weights_storage','input_storage','control'):cp('h2d',n,pe)
            launch('accumulate')
            if tile in (0,44):
                for pe in range(8):
                    for n in ('output_storage','state'):cp('d2h',n,pe)
        launch('finalize')
        for pe in range(8):
            for n in ('output_storage','bf16_storage','state','weight_guard_snapshot','input_storage','timing'):cp('d2h',n,pe)
        for fragment in FRAGMENTS:
            if fragment.identity==8:
                launch('preprocess')
                for n in (*QK_READS,'handoff_state'):cp('d2h',n,8)
            for pe in range(10):cp('h2d','handoff_control',pe)
            launch('arm_handoff');cp('d2h','handoff_state',fragment.destination)
            launch('handoff')
            for pe in (fragment.source,fragment.destination):
                for n in ('handoff_state','wire_data','wire_ack'):cp('d2h',n,pe)
            cp('d2h','qk_input' if fragment.destination==8 else 'attention_input',fragment.destination)
        launch('attend')
        for n in (*ATTENTION_READS,'handoff_state'):cp('d2h',n,9)
        for n in ('qk_input','qk_output','qk_weights','handoff_state'):cp('d2h',n,8)
        for pe in range(8):
            for n in ('output_storage','bf16_storage','weights_storage'):cp('d2h',n,pe)
        launch('release')
        for pe in range(8):cp('d2h','state',pe)
        for pe in range(10):cp('d2h','request_state',pe)
        for pe in (8,9):cp('d2h','handoff_state',pe)
    copies=launches=0
    for row in rows:
        if row['kind']=='copy':copies+=1;row['sequence']=copies
        else:launches+=1;row['ordinal']=launches
    return rows


def budget():
    seq=sequence();copies=[r for r in seq if r['kind']=='copy']
    return dict(copies=len(copies),launches=sum(r['kind']=='launch' for r in seq),
                host_bytes=sum(r['host_bytes'] for r in copies),payload_bytes=sum(r['native_bytes'] for r in copies),
                max_host_buffer_bytes=max(r['host_bytes'] for r in copies),fabric=fabric_budget())


def declared_bytes(role):
    own={'producer':PRODUCER_ARRAYS,'qk':QK_ARRAYS,'attention':ATTENTION_ARRAYS}[role]
    arrays={n:(v if n in own or n in COMMON else (1,v[1])) for n,v in ARRAYS.items()}
    extra=524 if role=='producer' else 0
    return dict(arrays=arrays,explicit_kernel_and_timestamp_bytes=extra,
                total=sum(n*b//8 for n,b in arrays.values())+extra,
                excludes='Code,padding,compiler state,memcpy allocations,stack; actual ELF end+4096<=49152 required on all10PEs')


def resource_plan():
    return dict(pes=10,producer_slabs=list(range(8)),consumer_pes={'qk':8,'attention':9},
                traffic=budget(),buffers={r:declared_bytes(r) for r in ('producer','qk','attention')},routes=routes(),
                leases={'producer':{'IQ':[2],'OQ':[2]},'qk':{'IQ':[2,3,4,5,6],'OQ':[2,3,4,5,6]},
                        'attention':{'IQ':[2,3,4,5,6],'OQ':[2,3,4,5,6]},
                        'shared_serial_transport':{'tasks':[10,11],'UT':[2],'DSR_dest':[5],'DSR_src1':[5]},
                        'producer_GEMV_DSR':[3,4], 'SDK_reserved_colors':[20,21,22,23],'SDK_reserved_queues':[0,1]},
                planning_estimate_seconds=1100,planning_hard_seconds=1200,
                estimate_basis='WP14 measured256.0076s for4 producers x1 token; this proposal8x2 gives4x contraction/readback work =1024.0304s. Allow75.9696s for additional10PE routing/attention/cache/metadata work. This additive allowance and scaling are unmeasured, not a performance guarantee. Controller may refuse before SDK admission.',
                supervisor='Original guard retains SDK300/CPU60. Separate guarded_wp15.py requires exact controller manifest admission, compile<=300 and simulate<=1200. Recipe preparation grants no SDK permission.',
                resource_limits={'RAM':20*1024**3,'swap':0,'available_RAM_reserve':8*1024**3,
                                 'cache':20*1024**3,'free_SSD_reserve':32*1024**3,'compiler_parallelism':1,'simulator_threads':1},
                planning_only=True,sdk_execution_authorized=False)
