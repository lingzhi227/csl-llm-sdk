# Full original K-width contraction

This component lowers the complete40,48 and136 K blocks used by the original
5120-,6144- and17408-column Qwen projections. Each leaf owns one original2x128
FP8 tile and scale. Qualification uses two output rows per projection: complete
input width is not a claim to execute every output row or the whole model.

## Static hierarchy and resource ownership

`spatial/contraction.py` recursively partitions a contiguous preorder interval.
The first PE owns the node's local dot; the remaining interval is split into
at most two contiguous child intervals, with the left interval receiving the
extra PE. Every node's subtree is therefore an exact ordered interval of K blocks.
For136 participants the maximum root-to-leaf depth is seven edges.

Each tree depth has two distinct colors, one per child side. Paths on the same
color have disjoint intervals because their parent subtrees are disjoint. The
child source injects from RAMP, intermediate routers follow the selected
neighboring-PE path, and the parent consumes at RAMP. Only the parent consumes the message; intermediate
PEs do not execute software relays. The plan checks route continuity and conflicting
route definitions. Source tests independently walk the resulting paths to their
expected endpoints and enumerate all disjoint subtree members.

At most two child input queues and one tree output queue are required per PE;
three words carry two FP32 sums and one exact participant count. A node waits for
its own native dot and both expected child completions, then adds local, left and
right values in that fixed order. Send completion releases its aggregate buffer.
The separate chain control uses alternating colors17/18 and a third input queue,
and reduces from highest to lowest K. It executes the same local dot work.
The plans use DSR3/5 for child receives, DSR6 for sends, and reserve DSR4 for the
native dot. Compiler/memcpy DSR0:2 remain free.

The default rectangular probe packs224 active matrix tiles into8x28 PEs with
no padding. Consecutive logical K ranks follow a serpentine nearest-neighbor path,
occupying disjoint8x5,8x6 and8x17 subregions. Manual geometry knobs also support
a4-wide snake and the initial straight136x3 padded layout. Source checks cover
all three mappings; execution evidence names its exact geometry. Full-K reduction
behavior is distinct from a full-model layout or full-M execution.
Weight banking, dynamic projection route selection and complete model ownership
are separate unimplemented admission requirements.

## Numerical and timing contract

Host preparation pins the original layer0 shard and reads every K column of the
selected gate, linear-attention output and MLP down-projection rows. Native local
dots must equal the independent FP32 emulator exactly, with zero signs canonicalized.
Every tree subtree and chain suffix has its own independently frozen ordered FP32
answer and FP64 absolute-product bound. Tree and chain use different reduction
orders; their outputs are not required to be bit-identical to each other. Each
must separately satisfy the unchanged numerical contract. Packet encoding,
participant counts, endpoint callbacks, original weights/scales and raw inputs
are also checked.

Four physical cases cover mixed values, zeros, tiny scales and large signed
values, with changing original output rows including scale-block boundaries and
matrix-end rows. Simulator coverage is fixed at the mixed case with both schedules,
covering all three full K widths and a changed scheduling mode without runtime
reset. Physical coverage remains all four cases in both modes. Cases and bounds
are prepared before any candidate result is observed.

Each row root measures its own start through complete full-K result. No timestamps
from different PEs are subtracted. Device work includes original-byte operand
encoding, weight decode, local native arithmetic, full reduction and launch-entry
skew. Before starting, the host arms receives and verifies all endpoints ready;
that separate wall interval is reported. Root numerical readiness can precede
some sender completion callbacks; the host checks every callback before rearming.
Input/weight transfer is outside device timing. Already-quantized activation
fixtures enter the component; P8 dynamic quantization is not silently included.
No component cycle result is converted to full-model tokens/s.

The static express-route design uses standard CSL color/queue routing, with
immutable routes during execution. This source is an independent implementation;
no research runtime was copied.


## Preserved attempt progression

`contraction-sim-001` compiled the initial136x3 layout and admitted all408 PEs at
12,432 bytes maximum including4,096 stack. Its run reached the240-second deadline
without a complete numerical record. The source, failure and SRAM result remain
frozen. The original driver lacked phase logs, so the exact stall/cost location
is not established. No hardware was submitted for that attempt.

Attempt002 changes the physical mapping to8x28 with224 active PEs, preserving
logical trees, full K widths, arithmetic, cases and bounds. It adds phase logs to
distinguish upload, arming, numerical completion and retention. This is a new
geometry/diagnostic attempt, not an unchanged retry or relaxed acceptance gate.

Physical001 successfully compiles and admits224 PEs, but runtime creation rejects
the artifact as absent before any neural execution. Its1,131,713-byte archive
crosses the earlier component uploader's1MiB chunk boundary. The runtime job is
cancelled and released by the supervisor; no numerical physical pass is claimed.
The unchanged archive/source is reused in physical002 with an explicit8MiB
single-message admission, following the prior full-model transport fix. A no-job
recording stub proves that this exact artifact now produces the same one protobuf
message as the installed original SDK, versus two under the prior bounded path.
It does not prove what happened inside the server on001. No recompilation,
arithmetic change, weight change or acceptance relaxation accompanies this repair.
The original functional runtime remains unchanged; component-specific wrappers
and their provenance are under `performance/runtime`.

## Qualified physical result

Physical002 completes all four cases in both schedules, with normal stop and
release. At40/48/136 K blocks, tree intervals are1965–1972 /1994–2000 /2274–2275
cycles versus chain5100–5102 /5815 /14389. Maximum compiled SRAM including4KiB
stack is12,432 bytes. All intermediate results, independent bounds and complete
retention checks pass. The same artifact was reused after the upload framing
repair; its SHA and every source/ELF identity are retained in the evidence.
See MILESTONES.md and `evidence/contraction-hw-002/summary.json` for scope and jobs.
The full original model and its2,000 dependent tokens/s target remain unachieved.
