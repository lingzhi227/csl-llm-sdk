# WP15: original projected selected attention, accepted two-token device scope

CPU reference001 and the ten-PE SDK candidate are independently accepted. Both
original-hidden tokens completed all numerical/transport/cache/retention gates and
normal shutdown. See the executed measurements and limits in [report](WP15-REPORT.md).
The source/resource recipe was frozen as a proposal before execution and remains
unchanged; its separate controller admission allowed one run. WP09 remains unresolved.

## Computation and ownership

PE0–7 compute the same eight128-row, full5120-column contractions as WP13.
Global output order is Q256,rawgate256,K256,V256. The original projection-input
hidden vectors are densecase0 atposition0 and changedcase1 atposition1, in the same
request generation1 and runtime. The host uploads only original hidden tiles,
selected original weight tiles, static trained norm/frequency/probe parameters,
initial guard/poison values and command metadata. CPU projections/normalized
vectors/attention probabilities never become candidate device inputs.

PE8 receives raw Q from slabs0/1 and raw K from4/5, then runs the unchanged
trained RMS256 and partialRoPE64 kernel. PE9 receives rawgate from2/3 and V from6/7.
Four additional128-word messages from PE8 write rotated Q256/K256 directly into
PE9's attention operand buffer. PE9 runs the unchanged attention256/cache8 kernel:
QK dot, BF16 dot, scale1/16, BF16 score, max-subtract, exp, denominator/inverse,
FP32 probability, BF16 probability, uncertain cached V reduction, BF16 attention,
sigmoid(rawgate), BF16 gate, final FP32 product and BF16 output.

This is one selected Q/KV head, two original projection-input vectors. It excludes
upstream hidden RMS/embedding, the other23 query heads and GQA sharing, output
projection, complete layers, generation, long context and physical WSE-3.

## Distinct identities and lifecycle

`request_control[4]` is contraction ID, request generation, zero-based cache/text
position, one-based token count. Proposed token tuples are(1,1,0,1),(2,1,1,2).
The separate GEMV `control[0]` and its accepted accumulator generation counters
carry the monotonic contraction ID. They do not reset attention request state.
`qk_control[3]` is request generation, text position, contraction ID.

Every PE owns `request_state[8]`: last contraction, request generation, next cache
position, phase, cumulative begins, releases, request initializations, error.
Phases are0 initialized,1 active,5 released. A begin requires the next contraction,
matching request and next position, capacity<8, and a released/initialized state.
`initialize_request` requires an increasing request generation and position/token0;
it does not reset the contraction counter. On PE9 it resets cache validity metadata
only. Physical cache bytes remain observable. The proposed run calls it exactly
once; beginning token2 must preserve the entire first-token physical cache.

`handoff_state[32]` slots are:

| Slots | Meaning |
|---|---|
|0–2|contraction,request generation,position|
|3|phase:0 idle,1 armed receive,2 sending/waitACK,3 completed transfer,4 computed,5 released,9 error|
|4–5|received fragment mask,transmitted-and-ACKed mask|
|6–9|data received,data sent,ACK received,ACK sent counts|
|10–11|received/sent data words|
|12|protocol error|
|13–15|handoff command arrived,transfer complete,command unblocked|
|16|monotonic local event ordinal within this contraction|
|17–20|data receive/send completion,operand commit,ACK completion,unblock event|
|21|local consumer computation count,required exactly1|
|22–24|reserved zero|
|25–27|current fragment,source PE,destination PE|
|28|reserved zero|
|29|handoff command arrival event|
|30–31|reserved zero|

Before each fragment's data movement the selected receiver arms its input queue.
Its handoff command and completed ACK send form a join, allowing receive-first or
command-first arrival. Event29 and receive/commit/ACK/unblock events are captured
at both endpoints; all24 receiver transfers were command-first. Receive-first remains supported by the join source but was not observed in this run. A source
sends data, waits for send completion, then arms ACK receive. It unblocks only after
checking the complete matching successful ACK. All transfers are serialized.

## Wire contract and placement

Each data message has140 raw32 body words, stored between two guards in142 slots:
magic0x50414441,version1,ten descriptor words,128 canonical low16 BF16 words.
The ten descriptor fields are contraction,request,position,source PE,destination
PE,fragment ID,tensor role,row offset,count128,destination offset. ACK has13 body
words between two guards in15 slots:magic0x5041414b,version1,the same ten fields,
status0. Guards are0xa55aa55a and0x5aa55aa5. The receiver validates the complete
frame and all payload high16 bits before changing any target word or received mask.
Stale identities,duplicates,missing fragments and role/offset aliases are refused.
Malformed-command recovery is outside the device proposal; host model rejection
checks are not a device fault-injection qualification.

| Fragment | Source → destination | Role | Row offset | Destination offset |
|---|---|---|---|---|
|0,1|PE0,1 → PE8|raw Q|0,128|0,128|
|2,3|PE2,3 → PE9|rawgate|0,128|768,896|
|4,5|PE4,5 → PE8|raw K|0,128|256,384|
|6,7|PE6,7 → PE9|raw V|0,128|512,640|
|8,9|PE8 → PE9|rotated Q|0,128|0,128|
|10,11|PE8 → PE9|rotated K|0,128|256,384|

PE8 needs received mask0x33 before preprocessing and transmitted/ACKed mask0xf00
before release. PE9 needs mask0xfcc and eight matching ACK-send completions before
attention. Producer values stay intact through all downstream work and observation.
All8 producers' complete resident weights and guarded FP32/BF16 outputs are read
again after attention, before release. PE8 input/output/weights and outgoing masks
are also rechecked after attention. Both tokens retain16 separate producer summaries in total.

## Routes and hardware leases

Use a10PE line. Link i originates at PEi; i=0,1,4,5 ends at8, i=2,3,6,7,8 ends at9.
Data color is2+i, ACK color11+i. Intermediate PEs route data WEST→EAST and ACK
EAST→WEST. Endpoints use RAMP injection/ejection. All18 colors2–19 are distinct;
SDK colors20–23 are untouched.

Each producer uses OQ2 for data and IQ2 for ACK. PE8 uses IQ/OQ2,3,4,5 for its
four incoming links(0,1,4,5) and OQ/IQ6 for outgoing data/ACK. PE9 uses IQ/OQ2–6
for links(2,3,6,7,8). An intermediate route reserves no queue. App transfer tasks
are10(receive completion) and11(send completion), UT2, destination DSR5 and
source1 DSR5. The unchanged producer GEMV uses DSR3/4. All app transfer directions
reuse their leases serially. SDK IQ/OQ0/1 and its command/task leases remain intact.
Static queue/color checks cannot establish compiler allocation or runtime behavior.

## Observations and numerical gates

The physical sequence is generated once from `core/qwen38/wp15_layout.py` and is
checked at every host copy/launch, with fsynced entry/exit records, symbol IDs,
shapes, byte counts, word hashes and stable handles. The static AST audit separately
enumerates the actual driver/helper control flow and compares all3201 operations
against the plan. It also checks39 symbols/types/array extents on each PE role.

Each producer retains FP32 prefixes after112 and5040 columns, full5120 FP32/BF16
results, exact RNE, source projection radii, guards, final input/padding, all46 timing
pairs and cumulative state. Every data/ACK frame is read at both endpoints and the
complete destination operand buffer after each fragment. QK reads include trained
parameters, all stats/stages/casts, probes, events, tails and exact zero/sign cast
behavior. Token1 position0 exercised128 zero products with exact sign/cast checks. Neither token produced a zero sum; cancellation coverage is unchanged.

Attention reads all dot/scale/shift/exp/probability and cast arrays, denominator/
inverse, V reduction, gate input/exp/sigmoid/product, final casts, full4098-word
physical cache including guards and unused slots, metadata and events per token.
Actual operands have separate conditional arithmetic gates for dot,scale,shift,
exp,sum,inverse,probability,V,gate and product, plus exact casts and full cache/frame
checks. The original-hidden source intervals are carried independently through
current Q,all cached K/V,current gate and final output. No observed match zeros or
shrinks an original projection radius.

The accepted token2 source final gate is broad and does not reject uniform
attention. A read-only witness from frozen CPU operands yields dot centers
-9.286867022514343 and-30.687396466732025 with allowed FP32 absolute errors
0.006588256173241887 and0.00599539864037429. Both reject the zero dots produced by
the frozen ignore-QK/zero-query counterexample. The driver repeated this conditional
witness using actual device Q and cache K; both zero dots were rejected on token2. This
witness complements the original-source gate; it is not a new source bound.

## Resource proposal and execution boundary

3050 copies,151 launches,43,897,600 host bytes,22,192,404 native bytes; largest host
allocation57,352bytes<=65,536. There are24 data and24 ACK messages,14,688 injected
fabric bytes and53,856 route byte-hops. These are logical traffic accounting, not
measured hardware utilization. Initial array sizes, one-element stubs on unused
PEs, kernel globals and timestamp arrays total32,004bytes per producer,
12,294bytes on QK,18,182bytes on attention. Code,alignment,memcpy/compiler state
and stack are additional: actual ordinary ELF end+4096 must be<=49152 on all10PEs.

Plan1100s simulation, proposed hard1200s. Four times WP14's measured256.007598589s
is1024.030394356s;75.969605644s is an unmeasured extra allowance for this layout,
attention/cache and extra observations. Versus four WP14 runs the planned native
traffic increases69,108bytes and copies234, while launches decrease85. Scaling
and extra allowance remain estimates, and the controller may reject the proposal.

A separate `guarded_wp15.py` requires an explicit controller admission matching the
entire frozen candidate manifest. It admits only compile<=300s followed by this
one simulate<=1200s. The original `guarded_run.py` SDK300/CPU60 limits are unchanged.
Memory20GiB,Swap0,8GiB available reserve,20GiB cache,32GiB free SSD,one global heavy
lock,CPU quota400%,one simulator/compiler thread,core disabled,64MiB file limit,
128 tasks and exact owned-unit cleanup remain unchanged. Completed units/logs and
failures remain evidence; no unchanged rerun. Preparing a recipe is not admission.
All roughly10MiB existing selected payloads are reused on workstation, with no
HTTP download, installation, GPU or new agent. Raw payloads never enter public GitHub.
