"""One-PE original-input composition; distinct immutable and derived buffers."""
from . import attention_protocol as attention
from . import qk_rope_protocol as qk

def attention_name(n):
    return n if n in ('control','state','cache') else 'attention_'+n

def qk_name(n):
    return n if n in ('control','state') else 'original_input' if n=='input' else 'qk_'+n

ARRAYS={attention_name(n):v for n,v in attention.ARRAYS.items()}
ARRAYS.update({qk_name(n):v for n,v in qk.ARRAYS.items()})
ARRAYS['original_input']=(1026,16);ARRAYS['control']=(3,32)
READS=tuple([attention_name(n) for n in attention.READS]+[qk_name(n) for n in qk.READS if n!='state'])
INITIAL=tuple([attention_name(n) for n in attention.INITIAL]+['attention_input']+[qk_name(n) for n in qk.INITIAL])
ATTENTION_EVENTS=[1,2,3,4,5,6,7,8,1,2,3,1]
QK_EVENTS=[1,2,3,4,5,6,7,8,0,0]
OVERFLOW_EVENTS=[1,99,0,0,0,0,0,0,0,0,0,1]
LEDGER={'shape':[1,1],'head_dim':256,'cache_capacity':8,'rotary_dim':64,'positions':[0,7],
 'app_colors':[],'app_async_tasks':[],'physical_host_buffer_limit':65536,'stack_allowance_bytes':4096,'ordinary_address_ceiling':49152,
 'successes':10,'overflow_calls':1,'initializations':2,'launches':13,
 'ownership':'original_input never overwritten; qk_output512 produced then device-copied to attention_input first512; original V/gate copied to remaining512; cache owns persistent K/V',
 'completion':'shared token commit/unblock only after QK, copy, attention and final casts; qk_events never signal independent host-command release',
 'overflow':'capacity refused before QK, data/stages/cache writes; only shared error/rejection metadata and attention control events change',
 'probes':qk.PROBES}

def plan():
    rows=[('h2d',n) for n in INITIAL]
    for call in range(1,11):
        if call in (1,9):rows += [('h2d','control'),('d2h','state'),('d2h','cache')]
        rows += [('h2d','original_input'),('h2d','control')]+[('d2h',n) for n in READS]
        if call==8:rows += [('h2d','control')]+[('d2h',n) for n in READS]
    return rows

def budget():
    rows=plan()
    return dict(calls=len(rows),host_slots=sum(ARRAYS[n][0] for _,n in rows),payload_bytes=sum(ARRAYS[n][0]*ARRAYS[n][1]//8 for _,n in rows),max_host_buffer_bytes=max(ARRAYS[n][0]*4 for _,n in rows),launches=13)

def expected_state(gen,pos,total,inits,rejects,initialized=False):
    return [gen,pos,total,inits,0 if initialized else pos+1,0 if initialized else 3,0,rejects,0 if initialized else 1,total,0 if initialized else pos+1,8,256,0,0 if initialized else 0xa55a,0 if initialized else 0x5aa5]

def validate_ledger():
    assert len(READS)==23 and len(set(READS))==23
    assert budget()==dict(calls=297,host_slots=159134,payload_bytes=390188,max_host_buffer_bytes=16392,launches=13)
    assert len(set(INITIAL))==17 and 'attention_input' in INITIAL
    return True
