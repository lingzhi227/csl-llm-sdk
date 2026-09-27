# Regional GEMV contract and P5 physical qualification

P3 qualifies arithmetic and P4 qualifies the small resident decode lease. The P5
executable boundary includes the operand producer/transport, local dot,
inter-PE sum and completion, with real original weights and same-PE timing.
A fast local dot does not establish this boundary.

The complete imported model has only three contraction widths in units of128:
40,48 and136 (370,64 and64 matrices respectively). All are divisible by8. This
permits an explicit eight-lane K-group abstraction, with5,6 or17 groups per output
row tile. The actual physical placement is unassigned; the compact cyclic bank
plan is a capacity candidate, not an already optimized route plan. Mapping its
linear IDs naively into a rectangle can put adjacent K tiles across a distant
row or wafer wrap. Such mappings must be checked and rejected or replaced by
local clusters before claiming locality.

The first bounded executable region is implemented in `spatial/regional.py` and
`probes/regional_gemv/`. Its checked contract is:

- Start with an8x8 region: K block per column, two output rows per row. Original
  rows0:16 and K0:1024 of a pinned projection form a precisely declared partial
  matrix fixture. It is a component experiment, never a reduced model run.
- Column producers multicast128 already encoded half operands plus one FP32
  activation scale (65 words). The producer's conversion must itself execute on
  device and agree exactly with the P3 independent bit oracle.
- An endpoint posts an asynchronous receive, then decodes its next resident
  weight slot. Both receive completion and decoded-slot readiness are required
  to run its dot. Their DSR leases must not collide; weight/operand/sum storage
  has explicit ownership through the final send completion.
- First use a declared ascending-K FP32 chain sum to qualify transport and
  numerical boundaries. Receive the incoming partial while computing the local
  dot; combine only when both are ready. A later tree variant changes the
  floating-point schedule and needs a separately frozen independent bound.
- Source/core/sink stages are timed separately where possible. Whole-region
  completion returns to the original timestamp PE. Report the predecode cost,
  receive wait, arithmetic and final completion scope without assuming clocks
  on different PEs are synchronized. Compare overlap against a serialized
  variant with the same inputs and transfers.

Manual controls belong in a checked plan: row/K tiling, region dimensions,
producer positions, bank slot order, colors, queues, DSRs, tasks, buffer count,
reduction order and prefetch distance. The generated CSL and resource table must
come from the same plan. Requalify actual compiled SRAM after communication code
is added. All model matrices, state actors and the complete sentence loop remain
required for final acceptance.

`regional-gemv-sim-001` and `regional-gemv-hw-001` now pass all four fixtures in
both schedules. On the physical8x8 region, every serialized case takes2,599 root
cycles and every overlapping case2,188, a15.814% reduction in region completion
cycles. All original-weight partial sums equal the independent ascending-term
FP32 emulator exactly and satisfy the independent FP64 bounds. See MILESTONES.md
for resource release and measurement scope.

This is exact encoding of already quantized FP8 input bytes, not dynamic FP32 to
FP8 quantization. `source_quantization=false` records that boundary. The direct
IEEE encoder in `csl/fp8_encode.csl` is a separate uncompiled candidate, with a
host-only midpoint/neighbor/random-input identity check. It is not part of the
qualified P5 artifact and does not yet qualify the scale/division convention.

The generated plan records manual colors, queues, tasks and DSR leases. DSRs0:2
remain available to compiler/memcpy; dot owns4. DSR5 receives either incoming row
partials or a final row result depending on role. DSR7 sends a row partial/result
and can be reused for a completion acknowledgement only after its send callback.
The six stream colors are independently checked for route collisions and paired
links. Thirteen source tests include independent route endpoint traversal.

P5's compiled per-PE SRAM including the unchanged stack reserve ranges12,288 to
13,232 bytes. It contains one weight tile per PE, not the full resident bank.
Simply adding the maximum bank's extra35,500 bytes would exceed the ceiling for
some roles (up to48,732 bytes before composition/alignment changes). This is a
warning from separate-component accounting, not a failed combined compile.
The next placement must use actual role budgets and repeat combined compilation;
it must not assume that two separately fitting components also fit together.
