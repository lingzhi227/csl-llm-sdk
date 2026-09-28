# P25: compiled regional projection-to-fusion transport

Both complete original gate/up region geometries now compile with their actual
resident native workers, original-K reductions, root projection paths and
cohosted fusion reception. GDN layer0 uses5226 PEs in78x67; attention layer3 uses
5264 PEs in56x94. Each region has33 fusion actors. Compiler012/013 independently
cover every coordinate and admit actual ELF sections plus4096 bytes of stack at
a maximum47712 bytes, leaving416 bytes. This is whole-region compiler admission,
not neural execution, complete MLP/layer admission or a throughput measurement.

Frozen `layer-fused-routes-003` checks4032 projected streams and139264 original
eight-row blocks across all64 layers. Only identical relative geometries reuse a
route search; each translated region is independently walked and checked against
its original owners. Matrix/scale/auxiliary addresses remain those of P23
schedule002. P24's local-left-right arithmetic trees remain unchanged. Closed
interval coloring reduces the gate/up tree to seven colors. Congestion-aware
static paths use the remaining PE/color slots without multiple receive directions,
color swaps, changing routes or software transit forwarding.

## Ownership and hardware backpressure

The current path copies each packet into a fixed output queue. Its DMA completion
releases the private source arena, while downstream queue capacity supplies
hardware backpressure. A following output can reuse that arena after its last
word was copied. The stream is single-source, lossless and ordered, with a
permanently bound output queue/color and one receive direction per router.
No per-block software consumer receipt is generated or imitated in this mode.
It halves the planned root/fusion stream count from126 to63 per region and removes
the original per-row credit fanout and its producer barriers.

This completion is **local transport ownership**, not sink consumption, stage
completion or request/state release. `audit[3]` records only that worker's local
row loop. The controller must still establish actual downstream completion and
epoch drain before resetting state/reusing a request slot. That connection is
unfinished. Queue/color reconfiguration would require a separate drain proof and
is not allowed by this transport contract. The installed SDK queue-flush example
explicitly distinguishes copying into the output queue from that queue becoming
empty; see `layer-fused-sdk-semantics-001.json`.

The earlier explicit-receipt implementation remains an optional source mode and
in its frozen compiles. Copy mode requires every participating native worker in
the region to use the same ownership rule. Compiler012/013 bind it on every PE.

## Two independently arriving projection streams

Fusion ownership is an overlay on the existing low-payload actor candidates; its
group assignment supersedes P23's round-robin scratch schedule, without moving
any weight or auxiliary page. Minimum-Manhattan matching assigns distinct actors
near producer columns. An actor handles consecutive original128-value groups
whose first row belongs to its root. Only the following root can supply the final
group's suffix. Static root queues select the boundary prefix or main actor.

A single group buffer creates head-of-line blocking: a later root's first block
could wait until the preceding actor finishes several earlier groups. The new
actor has a separate256-byte early activation buffer and two44-byte receive
packets. It applies the unchanged BF16 SiLU/multiply to incoming pairs, retains
an early suffix, and merges it when its last original group becomes active.
Quantization still waits for every one of the group's128 original values. No
global17408-value intermediate vector is materialized in the actor.

The independent token-flow test pauses producer0. A one-buffer negative control
blocks producer1 at its first packet. With the separate suffix buffer all other
roots can drain; resuming0 completes every original group once. A separate
bounded-FIFO ownership test destroys the source packet immediately after local
copy completion, delays consumption, and checks all67 packets remain intact.
These are protocol tests, not device or numerical qualification.

## Composed PE resources

Native operand capture owns IQ2/UT2/DSR2 until all original K slices are retained.
Only then may a cohosted actor reuse UT2/DSR2 for its suffix receive on IQ7, into
a separate buffer. Its main receive uses IQ6/UT7/DSR7. Native ordered child
reception keeps DSR3/UT3; native compute uses DSR4 and output DMA uses DSR6/UT5.
Each actor resides on a non-root PE with16640 bytes of original matrix payload.
The compiler admits the combined body at each actual coordinate, including actors
with different reduction-child counts and prefix/suffix boundaries.

The fused down-output API reserves UT4/DSR5 through a readiness notification and
local send completion; an early consumer acknowledgment cannot release the source
buffer or this send lease. In explicit-receipt mode projected receipts share that
send lease and retain independent buffers. The current copy mode eliminates those
receipt sends. `fusion_step`/`fusion_signal` are integration/admission endpoints;
actual down-input distribution and its fabric callbacks are still missing.

## Preserved failures and evidence

- Routes001 passed an abstract graph audit but used color swaps. Actual WSE-3
  whole-region compile011 rejected them. The installed SDK tutorial also says
  color swap is unsupported on WSE-3. These routes are rejected, not reusable.
- Routes002 retained per-block receipts and tried fixed-color paths; the search
  failed after83 streams. Routes003 changes both the path constraint and copied
  transport ownership and is the current static overlay.
- Backend010 rejected boolean-array initialization; corrected backend011 passes.
  Projection009 rejected conditional module import; corrected010 passes selected
  cohost roles, but does not admit the later copy mode or complete fabric.
- Projection008 is the earlier17-role explicit-receipt compiler result.012/013
  are the current full-region copy-mode results. Every attempt's exact source,
  failure/success and verified workstation release are retained.

Full ELF inventories/artifacts stay on workstation storage. Compact census
receipts hash each full inventory and independently check every SRAM record and
coordinate. No WSE job was submitted for P25. Numerical changes from P23/P24,
complete input/down/residual paths, stage drain, neighboring layers, full-model
sentences and the >=2000 aggregate generated tokens/s target remain unqualified.
