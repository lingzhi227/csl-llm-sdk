"""Three-PE recurrent/gated RMS protocol; static transit routes survive resets."""
import copy
from .recurrent_protocol import LEDGER as RECURRENT_LEDGER
from .resource_ledger import check
LEDGER=copy.deepcopy(RECURRENT_LEDGER)
LEDGER['application']['colors']=[2,3,4,5]
LEDGER['application']['input_queues']=[2,3,4]
LEDGER['application']['output_queues']=[2,3,4]
LEDGER['placement']+='; PE2 is a small gated RMS consumer with no recurrent state'
LEDGER['messages'] += [{'phase':4,'kind':'global FP32 y','source':0,'destination':2,'color':4,'words':131},
                      {'phase':5,'kind':'completed norm ACK','source':2,'destination':0,'color':5,'words':3}]
LEDGER['routes']={'color4':'PE0 RAMP->EAST, PE1 WEST->EAST, PE2 WEST->RAMP',
                  'color5':'PE2 RAMP->WEST, PE1 EAST->WEST, PE0 EAST->RAMP',
                  'lifetime':'static throughout runtime and resets; PE1 early token completion never tears down transit routes'}
LEDGER['queues']['0']['norm_input']=4;LEDGER['queues']['0']['norm_output']=4
LEDGER['queues']['2']={'input':4,'output':4}
LEDGER['prepare']='host arm_consumer command returns after PE2 posts full131-word receive; root/peer no-op. Only then host launches token. Consumer joins token-command arrival and ACK send completion, supporting either order; exactly one token unblock'
LEDGER['ordering']+=' -> root sends global y -> consumer input BF16 RNE/gated RMS/output writes -> ACK send -> root validates ACK then commits/unblocks'
LEDGER['mixed_root_receive']='after ACK only first3 words are phase5 identity; retained128-word body is earlier phase3 local output. This is not a single coherent frame; validate header and retained body separately'
LEDGER['source_reuse']+='; root reuses dead local-y send body for global y only after output reduction; consumer retains full input packet and immutable ACK through completion'
LEDGER['budget']={'sim_seconds':300,'stack_allowance_per_pe':4096,'ceiling_per_pe':49152,
                 'estimated_host_words_upper':125000,'planning_seconds':180,'basis':'WP06 measured81.7s plus small consumer readbacks and134 extra fabric words/token; not a bandwidth guarantee'}


def validate_ledger():return check(LEDGER)


def validate_state(states,events,generation,token,total,initializations):
    expected=[[generation,token,total,initializations,4,0,3,2,token,1,265,1],
              [generation,token,total,initializations,4,0,1,2,token,1,131,1],
              [generation,token,total,initializations,4,0,1,1,token,1,131,1]]
    order=[[11,1,2,3,4,5,6,7,10,11,8,9,0],
           [8,1,2,3,0,4,5,6,7,8,0,0,0],
           [6,2,3,4,0,0,0,0,5,6,1,0,0]]
    return states==expected and events==order
