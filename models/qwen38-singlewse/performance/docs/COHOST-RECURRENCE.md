# Complete recurrent state co-resident with original FP8 banks

P13 is a complete-head state component for the P12 atlas, not a complete Qwen
layer or model. Its numerical input is the unchanged synthetic preprocessed
Q/K/V/decay/beta fixture from the original recurrent-head experiment. Convolution,
gate nonlinearities, normalization, full matrix routes and token feedback are
outside this component. `gdn-bank-hw-001` passes the complete96-position state
sequence plus8 reset-replay positions on physical WSE-3, with original banks
retained and all owned resources released.

## State, banks and explicit phases

The 4 by 5 application contains a controller above a 4 by 4 state rectangle and
three passive PEs. Horizontal rank owns one 32-element key interval; vertical
rank owns one 32-element value interval. Together, all 16 PEs hold the full
128 by 128 FP32 recurrent matrix. Every state PE also retains 111 original FP8
2 by 128 tiles and their exact expanded scales. Representative banks reuse the
first 111 slots from each of the six frozen P10 banks, repeated across the 16
state PEs. These are actual original bytes, but not every corresponding P12
physical owner's final full-model contents.

The controller preloads eight packets and issues one at a time. Arrival triggers
local decay and a 32-key prediction. A four-shard horizontal chain forms the
complete prediction for each value interval. The row root computes the delta
and broadcasts it to the other key shards. All shards update their own FP32
state and compute their query-weighted output. The same ordered reduction then
returns four complete value intervals to the controller. Counts, phase tags and
absolute position tags accompany reductions. The next request waits for all four
results and the previous request's send callback. Each worker's send callback
grants its next receive phase. State persists across host-supplied batches.

The per-value sequence is native FP32 FMA over each 32-key block, followed by
`partial0 + (partial1 + (partial2 + partial3))` in FP32. Prediction and output
both follow that declared order. The independent FP64 recurrence is unchanged;
a separate FP32 emulator freezes the block order before execution. The change
does not assert bit identity to the original 128-key sequential sum.

The optimized worker replaces scalar element loops for partial addition and
delta formation with native vectors, preserving every tested FP32 bit. State
and native FP8 dot phases share a 1,548-byte request area under an exclusive
lease: the state phase owns all bytes; the dot phase owns input bytes 0–259 and
decoded-weight bytes 260–771. Native dots require the state loop to be stopped.
The original weights and recurrent state never alias this scratch. This saves
772 storage bytes without expanding or changing checkpoint values.

Original FP8 bytes are held in aligned 32-bit storage words and are cast locally
to the unchanged packed-16-bit decoder view. Upload and full readback also use
32-bit words, halving transfer wavelets compared with 16-bit transfers while
retaining all original bytes. This is transport packing, not a precision change.

## Resource leases and admission

| Worker resource | Owner |
|---|---|
| Input queue 2, microthread 2, DSR 3 | Tagged chain receive |
| Input queue 4, microthread 4, DSR 2 | Next recurrent request |
| Input queue 5, microthread 6, DSR 5 | Delta receive |
| Output queue 2 or 3, microthread 5, DSR 6 | One active chain/delta/result send |
| DSR 4 and 7 | Synchronous state math; FP8 math is a separate phase |

The controller has four receive microthreads 2–5 and send microthread 6. Queue
IDs and microthreads are separate leases. Sources remain valid through send
completion; no sender callback is claimed to prove general fabric quiescence.

Actual admission uses the unchanged 48,128-byte ceiling and 4,096-byte declared
stack allowance. A new component checker counts coordinates in ELF load
rectangles rather than equating ELF file count with PE count. The compiler shares
one passive image between two coordinates: 19 images cover all 20 PEs exactly
once. Missing, overlapping and out-of-rectangle coverage are rejected. This
checker reuses the previously qualified ELF rectangle encoding with an explicit
simulator/physical fabric expectation; the original functional checker is intact.

## Numerical gate and preserved attempts

The original reference archive remains pinned to
`1279ce5a8e7311ee04b777c20f861111003cec8e03bf18531abc440785221882`.
Every returned output and each batch's complete terminal state must exactly match
the frozen block-order FP32 oracle and satisfy the original propagated FP64
bounds. Reset replay, complete bank readback, native FP8 dots at slots 0/56/110
before and after the recurrence, and noninterference with state are required.
The physical profile specifies 96 positions plus eight replay positions. The
bounded simulator specifies eight plus four. Every output is checked; intermediate
states inside a batch are not copied to the host.

Attempts remain frozen:

- `001` stopped at the original-reference archive hash gate. Regeneration on the
  workstation produced a different archive; its cause is unproven. Later attempts
  copy the exact retained original artifact. No numerical bound was relaxed.
- `002` and `003` expose compiler requirements for a tuple parameter and a
  compile-time fabric packet extent. Corrections do not change arithmetic.
- `004` compiles within SRAM, then the old image-count check rejects 19 images
  for 20 PEs. Exact load-coordinate coverage replaces that invalid assumption.
- `005` passes the 8+4 numerical sequences but reaches the 240-second run budget
  before full retention and normal stop. Its stable simulated step is 14,445
  cycles and its maximum compiled footprint is 47,792 bytes including stack.
- `006` exposes a forbidden global pointer cast while introducing scratch reuse.
  The decoder pointer is instead formed locally in the exclusive dot phase.
- `007` passes the same sequences and before/after native dots with vector joins
  and shared scratch. Stable simulated step is 6,628 cycles; maximum footprint
  is 46,784 bytes including stack. It times out during full bank readback, so it
  is not a complete accepted simulation. `008` packs the unchanged weight bytes
  into 32-bit transfers and retains the full readback requirement.

The 14,445-to-6,628 comparison concerns component cycle records inside two partial
simulator runs. It is not a physical speedup, complete inference rate, or evidence
that the complete 2,000 tokens/s target is met. The next route integration gates
are described in [ROUTE-EPOCHS-NEXT.md](ROUTE-EPOCHS-NEXT.md).

## Accepted simulator and physical result

`gdn-bank-sim-008` completes all12 positions,192 native FP8 output values, full
retention and normal stop in228.232 seconds total under the unchanged2GiB,
no-swap,two-CPU and240-second-per-phase limits. Its service and shared lock are
released. The32-bit packing changes no recurrent cycle results or weight bytes.

`gdn-bank-hw-001` completes96 dependent state updates plus8 reset-replay updates.
All13,312 returned FP32 values and13 complete terminal states (212,992 state
values) exactly match the frozen block-order oracle. All satisfy the unchanged
FP64 bounds; maximum state error is6.842145550134404e-8 and output error is
8.644613996855455e-9.192 native FP8 dot values pass before/after checks, state is
unchanged by those dots, and every resident original weight/scale byte is retained.
Weights upload once. Eight state steps run autonomously per batch; the host
supplies subsequent preprocessed packets between batches.

Physical request-to-complete-output latency is6,755–6,758 same-controller cycles,
median6,756.5. Inter-step gaps are58 cycles. Timing includes request multicast,
local state math, both reductions, delta broadcast and all four returned output
shards. It excludes host batch loading/readiness, initialization and terminal
callback auditing, which are separate observations. Clock frequency is not
assumed, and no model tokens/s is reported.

All20 physical PE coordinates pass compiled SRAM admission. Maximum low-section
end plus the declared4,096-byte stack is46,784 bytes, leaving1,344 under48,128.
The executable includes state communication and callable native FP8 dots, but not
the full matrix-network protocol or the controller's future preprocessing code.
This is admission of the component, not all P12 roles or the complete model.

Compile `wsjob-upwa9jj4dzqh4blvddwxxq` and runtime
`wsjob-bmakemdtsppwztgevc6ycq` both succeed and release normally. Stage wall times
are41.374 and81.434 seconds including infrastructure work; provider-billed node
hours are not exposed. A fresh account/system audit finds no owned active job or
assignment.27 source tests pass. Full-model2,000 dependent tokens/s remains unmet.
