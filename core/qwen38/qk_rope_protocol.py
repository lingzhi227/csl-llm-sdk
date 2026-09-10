"""One-PE Q/K preprocessing observations and fixed bounded probes."""
import math
from .rms_numerics import bits,from_bits,f32
ARRAYS={'input':(514,16),'weights':(514,16),'frequencies':(34,32),'probe_inputs':(10,32),
 'stats':(10,32),'rms_stages':(1026,32),'normalized':(514,16),'trig':(130,32),'trig_casts':(66,16),
 'rotary_stages':(386,32),'product_casts':(258,16),'output':(514,16),'probes':(34,32),'state':(16,32),'events':(10,32),'control':(3,32)}
READS=tuple(n for n in ARRAYS if n!='control')
INITIAL=('weights','frequencies','probe_inputs','rms_stages','normalized','trig','trig_casts','rotary_stages','product_casts','output','probes')
PROBES=[0.,from_bits(bits(math.pi/4)-1),from_bits(bits(math.pi/4)+1),f32(math.pi/2),f32(math.pi),f32(3*math.pi/2),f32(2*math.pi),7.]
EVENTS=[1,2,3,4,5,6,7,8,0,1]
LEDGER={'shape':[1,1],'head_dim':256,'rotary_dim':64,'positions':[0,7],'heads':['q','k'],
 'probes':PROBES,'app_colors':[],'app_async_tasks':[],'physical_host_buffer_limit':65536,
 'ordinary_address_ceiling':49152,'stack_allowance_bytes':4096,'calls':10,'initializations':2,'launches':12,
 'runtime_trig':'device position*immutable official FP32 inv_freq; no host runtime trig',
 'signed_zero':'tail exact bits; arithmetic comparisons include zero numerically; all actual BF16 RNE casts exact, nominal source signed-zero differences reported'}
def plan():
    rows=[('h2d',n) for n in INITIAL]
    for call in range(1,11):
        if call in (1,9):rows += [('h2d','control'),('d2h','state')]
        rows += [('h2d','input'),('h2d','control')]+[('d2h',n) for n in READS]
    return rows

def budget():
    rows=plan()
    return dict(calls=len(rows),host_slots=sum(ARRAYS[n][0] for _,n in rows),
     payload_bytes=sum(ARRAYS[n][0]*ARRAYS[n][1]//8 for _,n in rows),
     max_host_buffer_bytes=max(ARRAYS[n][0]*4 for _,n in rows),launches=12)

def expected_state(gen,pos,total,inits,initialized=False):
    return [gen,pos,total,inits,0 if initialized else pos+1,0 if initialized else 3,0,0 if initialized else 1,256,64,7,0]+([0]*4 if initialized else [0xa55a,0x5aa5,0xa55a,0x5aa5])

def validate_ledger():
    assert len(PROBES)==8 and all(0<=x<=7 for x in PROBES)
    assert budget()['calls']==185 and budget()['max_host_buffer_bytes']<=65536
    return True
