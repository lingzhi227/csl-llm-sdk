# Arrival-driven resident projections and fused block credit

P24 implements the actual native projection worker, its complete-K reduction
lowering, and the projection-to-fusion packet contract. It does **not** execute
a connected layer. Input distribution, root-to-fusion physical paths, external
credit fanout, GDN/norm integration and numerical execution remain open gates.
The full64-layer model and2000 aggregate generated tokens/s remain unachieved.

## Producer arithmetic and physical reduction paths

`runtime/layer_projection.csl` retains the original native operands after their
arrival, computes local rows, consumes the complete left then right subtree,
rounds only the complete-K result to original BF16, and sends branch-major packed
projection values. Epoch, row iteration and original K-contributor counts are
checked at each reduction. No host-supplied partial projection stands in for a
predecessor. The exported arm operation is initialization scaffolding; the full
stage controller still has to supply it through the actual serving path.

`spatial/layer_routes.py` uses balanced inorder trees over each group's contiguous
serpentine PE interval. Root selection changes, but every native weight address
and K slice remains the P23 address. The root is centrally located in rank space.
Depth/side colors have disjoint express-route intervals, so there is no native
route rewrite or same-PE/color alias in these reduction paths.

The independent walker covers128 MLP regions,2752 contractions,502320 workers,
499568 tree edges and2122720 route entries across all64 layers. It follows actual
physical directions, checks every original K slice exactly once, validates bank
base/stride calculations, and enumerates139264 projected blocks into all original
136 group128 quantizers per layer. The maximum individual tree edge is68 hops.
Colors3..17 are used by reductions; ingress2, credit18 and egress19 are separate
interfaces. Those interface colors do not by themselves establish complete
ingress/egress routes or legality of their combined layer network.

The1845150416 reduction word-hops per complete model pass include the listed MLP
tree headers and all row iterations. They exclude other neural operations,
input/consumer routes and credit traffic. This is traffic accounting, not latency
or throughput. No hardware timing is inferred from it.

## Shared scratch and explicit PE resources

The512-byte native decode arena has three nonoverlapping lifetimes: initial
operand reception, native weight decode/compute, then child packets and outgoing
projection. A child receive is armed only after native compute returns. The left
subtree is added before receiving the right; after both are consumed, the same
arena becomes the outgoing packet. It cannot be reused for the next row until
both send completion and true consumer credit have arrived.

The sequential child DMA retains the declared local-left-right FP32 order and
uses one DSR/UT pair for both child queues. Active resource assignments are:

| Function | Input/output queue | DSR | UT |
|---|---|---:|---:|
| Initial retained operands | input2 | 2 | 2 |
| Ordered left/right packets | input3/input4 | 3 | 3 |
| Projection/reduction send | output2 | 6 | 5 |
| Actual consumer credit | input5 | 1 | 6 |
| Synchronous native arithmetic | memory | 4 | synchronous |

DSR5/7 and UT4/7 remain available for future fused composition. Availability is
not a proof that all added streams can overlap safely. The initial operand DSR
also becomes idle during the retained-input row loop, but reuse requires an
explicit lifetime guard. Physical liveness/backpressure is not yet qualified.

The single-K-slice specialization writes the native result directly into its
partial-output slot, avoiding the temporary vector and an extra zero add. Ragged
groups retain their multiple-slice accumulation. This new specialization and
the new full-K tree order require independent numerical qualification before
model acceptance; compilation is not that qualification. BF16 packing maps the
unchanged `qwen_math.round` function directly to packed u16 output.

## Fused consumption must release the small block

The root packet is three u32 tags (epoch, row iteration, first original output
row) followed by packed BF16 values for gate then up. An8-row gate/up packet is
11 words. `csl/layer_fusion.csl` checks its original quantization-group boundaries,
then directly applies the existing SiLU/multiply pair helper to the four pairs.

Once those values are copied/consumed into the fusion group's retained buffer,
the receiver produces the two-word epoch/iteration credit. It must not wait for
all128 values or the downstream packet: a producer waits for credit before
emitting its next8 rows, so deferring credit would prevent the rest of the group
from arriving. The credit buffer itself remains owned until its send callback.
Down fragments retain the separate send-completion plus downstream-consumption
condition. No new group can overwrite a pending projection receipt.

This method is compiled through the existing backend admission shell009. Its
actual network callbacks, routing among different roots/actors, and co-resident
native-worker communication are still to be connected. It is not a claim that
the root-to-fusion stream has already run on a device.

## Compiler and source evidence

`layer-projection-compile-007` admits16 selected role profiles from a census of
all502320 resident MLP workers. Each profile keeps its maximum actual original
matrix plus live auxiliary payload; unused tail capacity is not allocated as a
dummy full bank. Nothing is removed from the pinned schedule's tensor/state
assignments. Maximum ELF sections plus4096-byte reserved stack total47936bytes,
leaving192bytes under48128. These selected body classes do not replace a complete
every-instance layer compile: different constant specializations and additional
network/fusion code still need admission.

Failures are frozen:001 and003/005 exceeded SRAM;002 used an unsupported
`@lsr32` builtin;004 attempted a forbidden comptime pointer cast.006 admitted the
earlier two-child-DMA composition at48080bytes;007 subsequently admits the single
ordered child DMA with fewer resource leases. All original failed sources and
diagnostics remain available. The successful compiler checks include original
bank payload and declared stack; the dynamic stack peak has not been measured.

All102 source tests pass, including complete reduction routing, ragged K coverage,
exact runtime bank offsets including auxiliary bytes, cross-root group assembly
and rejection of missing output rows. No numerical runtime or new physical WSE
job occurred. Seven projection compile services and backend009 have verified
release receipts. The ALCF snapshot has no own active/assigned jobs; final billed
node-hours are not available from those fields.

Continue by lowering the still-open real operand and root-consumer routes,
binding credits to their actual group fanouts, then composing GDN/norm and the
complete layer0 -> layer1 execution. Preserve the original weights, two request
states and full model scope throughout that work.
