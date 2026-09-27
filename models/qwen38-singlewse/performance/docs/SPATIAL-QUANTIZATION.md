# Next spatial activation producer (not implemented)

P6's complete scalar group still takes about22,800 cycles, despite exact direct
encoding. The next producer should distribute the128 values over32 PEs (8x4,
four values per PE), retaining the original group scale and final packet contract.
Its output must connect directly to the65-word operand interface qualified in P5.
No host-produced FP8 values may stand in for that producer when integrated.

The proposed dependency chain is explicit:

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
this allocation is a proposal, not a verified artifact.

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
