"""Version-scoped named-resource admission for the reduced two-PE fixture."""
DEFAULT_MEMCPY={'colors':[20,21,22,23],'input_queues':[0,1],'output_queues':[0,1],
                'local_tasks':[21,24,27,28,30],'control_tasks':[33,34,35,36,37,40]}
LEDGER={
 'sdk_image_sha256':'fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
 'provider':'default <memcpy/get_params>, WSE3, width2 height1 channels1',
 'provider_contract_source':'reviewed SDK2.10.1 <memcpy/sys_params>',
 'provider_contract_source_sha256':'e70159b986fd92a4f4a83a0a46abe209ce64ed45903ac8ccda9ca0c7a4ef47df',
 'provider_reservations':DEFAULT_MEMCPY,
 'application':{'colors':[2],'input_queues':[2],'output_queues':[2],
                'local_tasks':[10,11],'control_tasks':[],'microthreads':[2]},
 'pe_roles':{'0':'rank1 partial receive -> join local GEMV -> SUM -> RMS/unit gain -> host unblock',
             '1':'local GEMV -> bulk send -> send-complete callback -> host unblock'},
 'routes':[{'color':2,'pe':1,'rx':'RAMP','tx':'WEST'}, {'color':2,'pe':0,'rx':'EAST','tx':'RAMP'}],
 'dsr_leases':[
  {'bank':'dest','index':3,'owner':'BF16 expansion','lifetime':'synchronous GEMV'},
  {'bank':'src0','index':3,'owner':'advancing BF16 weights','lifetime':'synchronous GEMV'},
  {'bank':'dest','index':4,'owner':'GEMV accumulator','lifetime':'synchronous GEMV'},
  {'bank':'src0','index':4,'owner':'GEMV accumulator','lifetime':'synchronous GEMV'},
  {'bank':'src1','index':4,'owner':'expanded weights','lifetime':'synchronous GEMV'},
  {'bank':'dest','index':5,'owner':'bulk destination','lifetime':'arm through receive/send callback'},
  {'bank':'src1','index':5,'owner':'bulk source','lifetime':'arm through receive/send callback'}],
 'async':'one fabric operation per PE; explicit UT2 on fabric operand; DSR bank+index5 owned until callback',
 'compiler_managed':'temporary DSR/stride/scalar/memcpy/math resources are compiler allocated, not claimed free by this ledger; preserve compiler allocation output',
 'ordering':'mode0 local-complete -> arm -> receive-complete; mode1 arm -> receive-complete -> local-complete; both -> SUM -> RMS -> exactly one root unblock',
 'ownership':'sender partial stable through send completion; receiver scratch stable through SUM; local partial stable through SUM; summed stable through RMS and readback; host starts next call only after both PE commands complete',
 'reset':'per-call flags/event counters reset before issuing work; call counter persists; kernel reloads all advancing DSRs, zeroes scratch/output; fabric extent128 reloaded each call',
 'completion_scope':'sender unblock is local-send done, not root normalization done; root unblock follows both inputs and normalization'
}


def check(ledger):
    app=ledger['application'];reserved=ledger['provider_reservations']
    for namespace in DEFAULT_MEMCPY:
        ids=app[namespace]
        if any(type(i) is not int for i in ids) or len(ids)!=len(set(ids)):
            raise ValueError('Invalid identifiers: '+namespace)
        if set(ids)&set(reserved[namespace]):raise ValueError('Provider collision: '+namespace)
    leases=[(x['bank'],x['index']) for x in ledger['dsr_leases']]
    if len(leases)!=len(set(leases)):raise ValueError('Duplicate explicit DSR ownership')
    return True
