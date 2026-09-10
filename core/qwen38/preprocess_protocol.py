"""Serial one-PE history preprocessing resource contract."""
from .resource_ledger import DEFAULT_MEMCPY,check
LEDGER={'provider_reservations':DEFAULT_MEMCPY,
 'application':{'colors':[],'input_queues':[],'output_queues':[],'local_tasks':[],'control_tasks':[],'microthreads':[]},
 'dsr_leases':[],'compiler_managed':'all arithmetic temporary registers/DSRs; no explicit fabric operations',
 'history':'384x4 BF16 persistent projected inputs; advance oldest-to-newest once per valid token; reset zeroes all entries and valid length',
 'weights':'384x4 BF16 immutable, uploaded once per runtime; asymmetric temporal coefficients',
 'observations':'conv and SiLU pre-cast FP32 plus BF16 bits, exp384, normalizedQ/K/scaledQ/V, gates and full history every token',
 'state':'generation,last_token,total_tokens,initializations,valid_history,phase,error,generation_unblocks',
 'command_phases':'token updates history/vectors and entersphase2; finalize materializes parameters, computes scalar gates, commits once and entersphase3; host only launches/observes',
 'ownership':'no overlapping command; input/weights immutable, history owned by device after initial guarded upload',
 'execution':'one PE, eight tokens; compile300/sim300; conservative simulator thread1',
 'sram':'actual highest application section end plus4096 <=49152; no use of configuration holes as capacity',
 'cost_estimate':{'host_words_upper':75000,'planning_sim_seconds':150,'basis':'WP07 standalone4x128 nonlinear calls took10.2s;8x384 plus history/state readbacks budgeted conservatively, not a throughput promise'}}


def validate_ledger():return check(LEDGER)
