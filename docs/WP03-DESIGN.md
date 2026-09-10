# WP03 — Persistent accumulation across the complete contraction dimension

The fixture selects output rows 0:128 of the original BF16 up-projection matrix.
Its 5,120 input columns are traversed as 45 tiles of width 112 followed by a tile
with 80 valid columns. This covers the contraction dimension for 128 outputs;
the original matrix has 17,408 output rows. It does not execute a full projection,
layer, model, or physical wafer.

## Ownership and lifecycle

The host verifies and packs bytes, schedules transfers and requests entry points.
One simulated PE owns the persistent FP32 output vector and every multiply-add.
`begin` validates a new generation and zeroes only the output interior. Each
`accumulate` requires the exact generation, total width, tile index, global offset
and valid count. It reloads kernel descriptors and preserves accumulated output.
`finalize` rejects incomplete work and commits that generation once.

Four calls share one runtime: bounded dyadic input, a different dyadic input,
one-hot at global column 5119, then zero after nonzero. Generation, completed tile
count, consumed columns and cumulative entry-point counts are checked. Host
negative tests reject duplicate, missing, stale and out-of-order submissions.
These tests are not evidence of executing malformed submissions on the device.

The final tile's unused weights are BF16 +1 and inputs FP32 +7. The kernel must
exclude those positions using the valid count, rather than depending on zero
padding. The kernel uses the inherited C01 DSR allocation (index 3 expansion and
index 4 FMA) with an extent determined by the tile's valid columns.

## Independent numerical contract

The CPU reference decodes the original row-major slab independently of the packed
device layout. Products of the selected BF16 weights and dyadic FP32 inputs are
accumulated using FP64 `math.fsum`. For a prefix of n columns, the FP32 forward
error bound is `gamma(n) * sum(abs(w*x)) + n * 2^-126`, with
`gamma(n) = n*2^-24 / (1-n*2^-24)`. This is the direct increasing-column FMA
contract, not a tree or per-tile partial-sum contract. Finite normal BF16 and zero
are admitted; nonfinite and subnormal weights are rejected before execution.

Readbacks after columns 112 and 5040 check intermediate results and device state.
Final results cover all 5120 columns. The last-column one-hot must equal the exact
decoded original BF16 column, and the final zero call must produce numerical
zero; signed zero is allowed. Each intermediate and final gate is mandatory.

## Observation scope and timing

Every call checks input/output guards and a finalize-time snapshot freshly read
from the actual weight-buffer endpoints. The snapshot includes the generation.
Source-ordered snapshot/commit markers are not independent timestamp evidence.
Only the fourth call reads the entire final resident weight tile. Consequently,
device weight immutability is checked for that final resident tile; no claim is
made about every historical tile. Frozen host slab and tile hashes are checked
before and after the run.

Each tile stores raw start/end counter words around the kernel only: three
little-significance-first u16 words per 48-bit counter. Elapsed cycles use modulo
2^48 subtraction. Host lifecycle durations can include queued simulator work,
transfers and observation overhead; they are not isolated hardware bandwidth.

## Bounded experiment plan

A preliminary 304-column diagnostic uses 112+112+80 and the same four call shapes.
Its measurements support a separately bounded full-width run. The full run keeps
all four calls and reduces redundant whole-weight readbacks to fit the budget.
This observation change is not arithmetic acceleration.

The supervisor admits one serial heavy job, enforces MemoryMax 20 GiB and zero
swap, and requires 8 GiB available RAM and 32 GiB disk reserve. Cache use is capped
at 20 GiB by preflight and periodic checks, not a filesystem quota. Compile and
full simulation each have a 300-second deadline. No failed candidate is retried
unchanged. Raw weights, compiled output and detailed result arrays stay outside
the public source tree.
