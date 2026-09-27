# Spatial activation producer and original-weight consumer

P6's complete scalar group still takes about22,800 cycles, despite exact direct
encoding. The implemented producer distributes the128 values over32 PEs (8x4,
four values per PE), retaining the original group scale and final packet contract.
It uses the65-word operand interface qualified in P5 and directly invokes one native2x128 original-weight consumer on the root.
No host-produced FP8 values may stand in for that producer when integrated.

The dependency chain is explicit:

1. Each PE owns four finite FP32 inputs and computes their absolute maximum,
   clamped from below by the same FP32 value1e-10. Since finite positive IEEE bits
   preserve numerical ordering, a checked unsigned maximum may implement the
   reduction without changing its result. Nonfinite inputs remain rejected.
2. Combine maxima along the restricted row/column tree. The root computes exactly
   the original FP32 multiplication by1/448, then broadcasts that one scale.
3. Each PE performs the same FP32 division and P6 encoding on its four inputs.
   Convert the resulting FP8 codes to the exact scaled-half representation already
   qualified by P3. This yields two packed32-bit words per PE.
4. Gather those words in fixed row-major input order. A node's buffer contains its
   own words, its east subtree, and (only in column0) its south subtree. Static
   subtree lengths define disjoint receive slices. The root receives all64 words
   and appends the scale as word64. Root completion means the complete operand
   packet is resident and ready for a downstream GEMV consumer.

Use separate max and payload planes, with their own callbacks and buffer credits.
A concrete initial resource allocation is max input queues2/3, scale input4,
payload inputs5/6; max/scale/payload output queues2/3/4. Reserve DSR0:2 for compiler
and memcpy; use3 for max transmit,4 for scale receive/transmit by role,5/6 for
payload receives and7 for payload send. Row/column alternating max colors3:6,
scale broadcast7 and alternating payload colors8:11 avoid overlapping routes.
The checked plan, actual CSL and compiled resource gate must agree before launch;
`spatial/quant.py` checks this restricted allocation and lowers the routes and per-PE parameters. Execution admission is recorded per immutable attempt below.

All receives are posted before a producer can deliver the corresponding phase.
Scale readiness gates division, own encoded data and all child payloads gate the
send, and send completion gates buffer reuse. Repeated epochs must verify complete
reset, unchanged inputs, deterministic placement and every endpoint's contribution.
Only same-PE root start/end time can report whole-producer latency without clock
synchronization. Include maximum reduction, scale distribution, encoding and full
packet gathering, plus any host-launch entry skew; do not count only parallel
local encoding.

Use the frozen P6 group vectors and independent PyTorch bytes/scales, including
midpoint neighborhoods, zeros/tiny values and seeded mixed groups. Verify every
rank's four outputs and the complete65-word root packet against an independent
packing oracle. A later complete-model allocator must preserve the128-value
quantization groups while mapping prior outputs to these input owners. That
placement, full model integration and2,000 dependent tokens/s remain unproven.

## Arming, measurement and consumer scope

The host first launches `arm` and checks all endpoint readiness records before
launching `start`. This guarantees the maximum tasks and asynchronous receives
are initialized; launch dispatch alone is not used as an all-PE readiness proof.
Arming and that readiness audit are separately timed on the host and excluded
from device producer cycles. A future self-driven model must integrate arming
with its producer/consumer credit protocol; this probe is not host-free token
execution.

The root timestamps complete gathered packet availability, then producer readiness
including its scale-send completion credit. It next invokes the exact native dot
without any host read or write between quantization and computation. Root DSR4
is leased to scale transmission first, and to dot computation only after its
completion callback. The constant original two-row layer0 gate weight tile is
decoded once before the measured groups; its preparation cycles and host time
are separate. This is one partial128-term consumer, not a full projection or
resident weight-bank execution.

The complete physical fixture remains the frozen44 P6 PyTorch groups. Simulator
coverage is fixed beforehand at the12 boundary groups and indices12/20/28/36.
Every local code, every scale bit and every subtree's ordered packet are checked,
including sentinel tails. The native output must equal an independent ordered
FP32 emulator and satisfy the separately frozen FP64 absolute-product bound.
Inputs, original packed weights/scales and endpoint event counts are retained.
No acceptance tolerance is chosen from the observed candidate outputs.

## Qualified P8 evidence

Simulator `spatial-quant-sim-001` passes16 predetermined groups; physical
`spatial-quant-hw-001` passes all44. The physical median producer-ready interval is
2,484 cycles, one native consumer519 cycles, combined3,003 cycles. Packet-present
median is2,422 cycles. Maximum compiled SRAM plus4,096 stack allowance is14,272
bytes. Both physical jobs succeeded and released normally; see immutable evidence
and MILESTONES.md for complete timing scopes and host costs. There is no complete
projection/model or autonomous inter-group dispatch in this result.
