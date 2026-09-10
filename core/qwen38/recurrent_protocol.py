"""Serial full-head recurrence transport/state contract; no replay recovery."""
from .resource_ledger import DEFAULT_MEMCPY, check as check_resources
LEDGER={
 'provider_reservations':DEFAULT_MEMCPY,
 'application':{'colors':[2,3],'input_queues':[2,3],'output_queues':[2,3],
                'local_tasks':[10,11],'control_tasks':[],'microthreads':[2]},
 'placement':'PE0 owns key rows0:64; PE1 owns64:128; each64x128 FP32 persistent state',
 'frame':{'elements':131,'element_unit':'32-bit word','header':['u32 generation','u32 token','u32 phase'],
          'body':'128 FP32 values transported as raw u32 words'},
 'messages':[{'phase':1,'kind':'prediction','source':1,'destination':0,'color':2},
             {'phase':2,'kind':'delta','source':0,'destination':1,'color':3},
             {'phase':3,'kind':'output','source':1,'destination':0,'color':2}],
 'queues':{'0':{'input':2,'output':3},'1':{'input':3,'output':2}},
 'dsr_leases':[{'bank':bank,'index':index,'owner':'synchronous kernel' if index<5 else 'single in-flight transport'}
               for bank,index in [('dest',3),('src0',3),('src1',3),('dest',4),('src0',4),('src1',4),('dest',5),('src1',5)]],
 'ownership':'DSR5 and UT2 held through local transfer completion callback; send source immutable until send callback; receive header/body consumed only after all131 words arrive',
 'ordering':'prediction reduction -> delta distribution completion -> local update -> output reduction; no concurrent fabric operation on a PE',
 'source_reuse':'root prediction packet becomes delta after prediction sum; delta becomes local output only after send completion and all local state updates consumed it',
 'reset':'only after both PE commands completed; new generation zeroes state and token counters, total commit count persists',
 'unsupported':'replay recovery, overlapping tokens, packet retransmission; wrong generation/token/phase is rejected, no claim of coordinated error recovery',
 'compiler_managed':'temporary DSR/stride/scalar/memcpy allocations remain compiler-managed; explicit3/4 kernel and5 transfer leases are separate'
}


def validate_header(words,generation,token,phase):
    if len(words)!=131 or any(type(x) is not int or not 0<=x<2**32 for x in words):
        raise ValueError('Expected131 unsigned32-bit words')
    if words[:3]!=[generation,token,phase]:raise ValueError('Wrong generation/token/message phase')
    return True


def validate_state(states,events,generation,token,total,initializations):
    if len(states)!=2 or len(events)!=2:raise ValueError('Two PE observations required')
    expected_events=[[9,1,2,3,4,5,6,7,8,9],[8,1,2,3,0,4,5,6,7,8]]
    expected_states=[[generation,token,total,initializations,4,0,2,1,token,1,262,1],
                     [generation,token,total,initializations,4,0,1,2,token,1,131,1]]
    return states==expected_states and events==expected_events


def validate_ledger():return check_resources(LEDGER)
