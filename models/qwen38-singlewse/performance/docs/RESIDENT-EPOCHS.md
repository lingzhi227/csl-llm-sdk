# Arrival-triggered resident epochs

P11 moves the P10 mixed resident actor from a host-triggered contraction to a
device-controlled regional loop. A2x4 rectangle contains six full112-FP8/12-BF16
bank workers, one controller and one passive PE. The controller sends a66-word
request through static2D multicast. Its header names precision, slot and fixture
case; the remaining65 words retain the existing native-dot operand contract.

Request completion activates the worker task. It resets that epoch's flags,
arms the two child receives, dispatches the local typed dot and joins local/left/
right results in the frozen order. A four-word return holds two FP32 sums, the
exact subtree count and a monotonic epoch tag. Every join checks the child tags
against its own epoch. The root sends to the controller through a separate route.

Only the aggregate's send-completion callback advances a worker and arms its next
request receive. This protects the SRAM source buffer from early reuse. A send
completion does not assert that an output queue has drained; routes, queue/color
bindings and packet order remain fixed throughout this component. The controller
issues the next request only after both its prior send and root-result receive
complete. Fabric backpressure can wait for a late worker's receive credit.

The host initializes all banks, requests and validation oracles once, then performs
one readiness audit before each96-epoch loop. It makes no calls inside the loop.
Afterward it reads the controller history and checks every worker's final callback
counts. Two runs start at different fixture indices to qualify reset and ordering.
The final audit covers callbacks that may trail the last root's numerical result.

## Explicit hardware resources

The request receive uses DSR2, left/right child receives use DSR3/5, and aggregate
send uses DSR6. Native arithmetic retains DSR4 and BF16 expansion DSR7. Installed
SDK2.10.1 memcpy sources declare destination/source1 DSR0 by default; no shared
SDK code is modified. These are separate namespaces from microthread identifiers.

Workers explicitly assign microthreads2/3/4/5 to left/right/request receives and
aggregate send. Controller send and result receive use microthreads2/3, despite
their distinct input/output queues both having ID2. The compiler-facing plan
checks collisions and exposes these assignments as manual parameters. WSE-3
permits explicit microthread assignment independent of queue ID; concurrent
operations must not share one. See the official
[WSE-3 microthread example](https://cerebras-sdk-docs-140.netlify.app/csl/code-examples/tutorial-topic-15-wse3-microthreads)
and [microthread semantics](https://cerebras-sdk-docs-110.netlify.app/csl/language/microthreads_wse3).

Attempt001 omitted the explicit assignments. It passed compilation/SRAM but the
simulator stopped at controllerP4.4 with `trying to term ut_instr[2]`. Source
inspection identified simultaneous send/receive operations defaulting toUT2.
Attempt002 assigns independent microthreads, retains the same arithmetic and
hash-identical fixture, and passes the complete suite in140.875 seconds. The failed
source, diagnostic and remote raw log are preserved; no hardware ran that failure.

## Numerical evidence and scope

The original744 bank tiles are reused byte-for-byte from the publisher-verified
P10 fixture. Twelve new shared operand patterns repeat eight times per run, with
alternating FP8/BF16 slots, zeros, tiny normal values and large patterns. A frozen
independent emulator supplies each PE with192 bytes of exact local/subtree results
for validation only. Those values do not enter neural arithmetic. Every expected
subtree is checked against its independent FP64 absolute-product bound before
dispatch. The device then checks all4,608 local/subtree FP32 values over192 epochs;
zero signs are canonicalized by floating equality. Error records, exact callback
counts, the entire tagged root history, reset/replay and all banks/oracles/requests
are checked at completion. Root histories also receive direct host FP64-bound checks.

Actual simulator SRAM reaches47,792 bytes including the unchanged4,096-byte stack
allowance, leaving336 bytes below48,128. Validation code and oracle storage are
included; no production-footprint reduction is assumed. No capacity was lowered.

`epoch-bank-hw-001` passes the identical192-epoch suite on physical WSE-3,
including all4,608 perPE checks, every epoch/count tag, both root histories and
all resident bytes. Physical SRAM is also47,792 bytes including stack. FP8
request-to-result intervals range1,689–1,703 cycles (median1,694); BF16 intervals
range1,659–1,660 (median1,659). The controller gaps are89–90 cycles. Complete96-epoch
loops take169,455 and169,474 cycles, or1,765.156 and1,765.354 cycles per epoch.
No comparison with P10's shorter root-only timing boundary is a measured speedup.

Compile `wsjob-xdtlrbvuw59bp9svown5yh` and runtime
`wsjob-rq9kf3ruklkqnnaalifpga` both succeed and release normally without cleanup
errors. Stage wall times are41.533 and198.691 seconds, including cluster setup;
provider-billed node hours are not available. The runtime's cluster status moved
from initializing to running before device execution; initialization time is not
kernel latency. Host start through the96-result history takes2.496ms and1.616ms,
including SDK command/transfer overhead. The separate final callback audit takes
about0.78ms and0.72ms. Fresh job/system accounting finds no owned active job or
assignment, and both bounded workstation services are terminal with the lock free.

Each timestamp pair comes from the same controller, from issuing a request through
receiving its numerical result. It includes multicast, resident dispatch, native
arithmetic, tree reduction and return. The full-loop interval additionally includes
the real gaps between epochs. Initial host readiness and final callback auditing
are separate; no cross-PE clock subtraction or nominal clock conversion is used.

Control advances depend on prior completion, but the preloaded neural inputs are
independent. This is neither autoregressive model inference nor a fullK/fullM
projection. The next integration boundary is complete-model physical bank ownership,
compact per-operation slot addresses and compatible routes, followed by connected
nonlinear/state operators. Full original sentence generation at2,000 dependent
output tokens/s remains the acceptance target.
