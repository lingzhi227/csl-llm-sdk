# Connected frontend and recurrent dataflow

P47 connects original convolution, Q/K normalization and gates directly to
recurrent state workers, then returns their actual outputs to gated normalization
on WSE-3. Original projected BF16 values are the diagnostic input boundary; final
gated values are the output boundary. The output projection, complete resident
layer and model conversation remain unqualified.

Physical `gdn-fusion-hw-001` passes all16 frontend groups,48 heads and753 original
state slices over six continuous plus two reset positions. A fresh audit checks
6,291,456 FP32 state observations,148,224 frontend FP32 values,49,152 core and
49,152 gated BF16 outputs. Histories, shared Q/K and bitwise reset replay pass;
all43,104 original parameter words remain unchanged. Maximum state relative L2
is6.447583e-7 and maximum error reaches0.165468 of its propagated element bound.
Observed packets and core values are verification inputs only; neither is sent
back to the device. The host supplies no recurrent result.

All800 diagnostic PEs/773 ELF images pass actual SRAM admission at a maximum
45,440B including4,096B stack. Thirteen workstation attempts and both cluster
jobs have released. The final account audit reports no active or assigned jobs.
See [summary](../evidence/gdn-fusion-summary-001.json) and
[fresh numerical audit](../evidence/gdn-fusion-hw-001/numerical-audit.json).

## Broadcast and return

`spatial/gdn_fusion.py` composes the arithmetic without changing its state
equations. Each group occupies a short serpentine path in a ten-column block of
the 2D mesh. This is a diagnostic geometry, distinct from P45's resident-bank
layout. Production routing and complete cohost resource admission remain open.

A group sends its three387-word head packets in order. Hardware counter filters
pass only each worker's own387-word window in the1161-word period. Other words
continue through the fabric without CE receive/discard tasks. The counter wraps
between positions while the recurrent state remains resident.

Each worker emits paired five-word frames, followed by a control wavelet. Its
return router initially receives from RAMP; the terminal control advances it to
receive from its child. `pop_on_advance` consumes the local ADV exactly once.
Ancestors see NOP and forward the child's frames entirely in hardware. The
frontend counts both data frames and worker markers. Only after both retire may
the next header arrive and reset the worker's return switch.

There is one receive direction per switch position. Return color4 supports the
WSE-3 switch, and input color5 carries the counter filter. These assignments do
not establish compatible colors or queues in the production bank layout.

## Separate lifetimes

A frontend marks the head eligible for return before launching its packet;
packet storage remains leased until the local send callback. Output retirement
can precede that callback. Final frontend retirement waits for both. Every group
in the qualification intentionally delays the last source callback until its
gated output completes, exercising the ordering directly.

Workers receive headers through a data task, then block it while bulk V/K/Q
receives and arithmetic own the decoder and descriptors. The task is enabled
after terminal-control send completion. Idle workers leave no bulk DMA pending.
Model state persists until an explicit reset after the complete graph drains.

SDK row command streams are independent. The driver first observes actual
frontend completion, then issues a fresh worker snapshot and checks local
retirement. Merely queuing `inspect` behind `await_done` on the frontend row
allowed earlier snapshots on other rows. Production scheduling must supply
readiness and scratch-lifetime dependencies on the device.

## Numerical and development evidence

Every produced frontend packet satisfies an a priori interval from the original
parameters. Independent FP64 recurrence consumes the observed FP32 packet for
verification and propagates FP32 error bounds over all prior states. All state
values and BF16 core outputs are checked. Gated normalization has its own
interval; convolution histories and reset replay match exactly.

Attempts001–011 preserve compiler errors, admission problems and transport
failures. Attempt009 completed192 core frames but lost terminal markers because
`always_pop` consumed control fields at every hop. Attempt010 corrected the
route, then expired during unlabelled numerical readback. Attempt011 proved
readback progress but rejected old cross-row state snapshots. The corrected012
passes two persistent positions and parameter retention. It transfers only live
state columns, avoiding unused diagnostic bank tails. Its900s simulator budget
is based on measured per-position capture cost, not wafer compute latency.
Full013 compiles every original slice with the same kernels and driver.

[Development receipts](../evidence/gdn-fusion-development-summary-001.json)
retain all attempts. Twenty-five source and25 publication regressions pass.

## Measured component timing

Timestamp differences are taken on the same PE and handle48-bit counter wrap.
They include instrumentation, diagnostic packet capture and the deliberately
late callback. Worker return time includes frame formation, queuing and sending.

| Physical interval | Cycles |
| --- | ---: |
| Frontend first head launch to complete group drain |227,519–250,397|
| Median over128 group/position observations |230,760|
| Worker key/state update |5,643–21,028|
| Worker query reduction |2,247–8,399|
| Worker result-to-terminal-control completion |420–138,991|

Runtime initialization takes137.904s; host transfers, complete state capture and
verification follow. These are reported separately from PE cycles and cannot
be treated as inference time. The substantial return/consumer serialization
motivates cross-kernel fusion and direct downstream consumption.

Next lower the connections into the complete original banks, checking routes,
queues, descriptors, microthreads, stack and temporary storage at every cohost.
Connect original output projection and retained normalization/MLP, then all64
stages and real token feedback. The acceptance target remains correct continuing
multi-turn dialogue at>=2000 average output tokens/s, including new-input
processing and first-token delay. This component result does not meet it.
