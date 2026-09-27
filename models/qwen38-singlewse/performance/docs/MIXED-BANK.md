# Mixed FP8/BF16 resident computation

P10 composes both arithmetic paths with a static reduction tree and the maximum
candidate per-PE occupancy:112 original FP8 tiles and12 original BF16 tiles.
Every tile has2 output rows and128 input columns. FP8 uses256 original bytes plus
one exact FP32 expansion of its original BF16 scale; BF16 uses512 original bytes.
The executable retains the qualified IEEE-half FP8 format. It does not switch the
program-wide16-bit arithmetic format to BF16.

`csl/bf16_dot.csl` streams each column's two BF16 weights into the high halves of
two zero-initialized FP32 temporaries. DSR7 owns that stream. DSR4 performs native
ordered FP32 FMA and is shared sequentially with the qualified FP8 dot. P9's child
receive descriptors3/5 and send descriptor6 remain independent, including while
child receives are pending. The original functional BF16 sources are preserved;
`probes/mixed_bank/REUSE.json` records their hashes and the adaptation.

The combined component removes P9's activation decoder and chain comparison.
The caller supplies the existing65-word operand contract:128 half-encoded FP8
values plus a scale, or128 raw BF16 words plus an unused sentinel word. It also
supplies an explicit per-PE4-byte `[kind,slot]` control packet, checked before
arming receives. There is no unused per-tile descriptor allocation. This changes
the metadata policy from the early bank capacity estimate. Complete-model address
generation, ownership, upstream quantization and device epoch control remain to
be implemented; the host fixture does not provide those capabilities implicitly.

`spatial/mixed_bank.py` lowers a2x3 serpentine tree containing all representative
node types:root/two-child, interior/two-child, interior/one-child and leaf. It
checks the resource leases and fixes the112/12 capacities. Six participants test
the combined executable footprint and lifecycle. They do not replace P9's separate
full40/48/136 K qualification or establish a complete matrix contraction.

## Frozen checks

The fixture contains744 original tiles from layer0 FP8 gate/out/down projections,
BF16 `lm_head`, untied embeddings and recurrent input a/b projections. Both original
shards are publisher-hash checked before bounded row reads. The original tensor,
row, column and resident slot are recorded for every tile. No full checkpoint
payload or compiled binary is published.

All banks are uploaded once. Twelve epochs execute first/middle/last slots, signed
zero, tiny normal and large boundary-pattern inputs, switches in both directions,
repeat execution after switches, and mixed types across PEs. Every native local
result and every subtree must match an independent frozen ordered-FP32 emulator,
with zero signs canonicalized. Every aggregate must also satisfy a separately
computed FP64 absolute-product error bound. Counts, all completion callbacks,
control/operand retention, exact replay, and every byte of all112/12 slots are
checked. BF16 subnormal arithmetic and embedding lookup are outside this suite.

The simulator attempt001 fails at compilation because a module alias shadows the
CSL builtin BF16 type. Attempt002 only renames that alias, reuses the identical
hash-pinned fixture after preparation-source checks, and passes all12 epochs,
full bank readback and normal stop in111.014 seconds. Both attempts remain frozen.
The maximum actual footprint is47,472 bytes including the declared4,096-byte
stack, leaving656 bytes below the unchanged48,128-byte ceiling. No full-model
SRAM admission follows from that component result.

## Physical result

`mixed-bank-hw-001` passes the identical12-epoch fixture on one physical WSE-3,
including every local/subtree FP32 value, FP64 bound, replay, completion callback
and full-bank retention. The maximum absolute FP64 error is0.00390625 in the
large BF16 boundary-pattern case; its frozen bound and exact FP32 gate both pass.
The sixPE root measures1,252 cycles for the FP8 cases,1,213 for the BF16 cases,
and1,217–1,245 for mixed-PE cases. Actual hardware-compiled SRAM is also47,472
bytes including stack. This is a representative bank/tree component result.

Compile `wsjob-cgptn3ebb4wdhvr3grwbdp` and runtime
`wsjob-jfpmomx3qw4gztsme62zh4` both succeed and release normally, with no cleanup
errors. Stage wall times are41.451 and81.573 seconds, including infrastructure.
These are not provider-billed node hours. Fresh account/system accounting reports
no owned active job or assignment; the workstation service and heavy lock are
also released. Other users' assignments are untouched.

## Timing boundary

Root timestamps surround typed resident dispatch, weight decode/streamed expansion,
the local native dot and complete numerical tree result. They include launch entry
skew. Initialization, operand/control upload and host receive-readiness auditing are
reported separately. Some sender callbacks may finish after the root result; all
are checked before the next epoch. There is no cross-PE clock subtraction or assumed
clock frequency. Comparing FP8 and BF16 cycle counts is not a precision speedup
claim:the paths consume different original weights and operand representations.

The remaining integration step is a device-controlled epoch with input arrival,
typed slot dispatch and completion credit, followed by all-matrix physical bank
ownership and routes. Those additions must repeat actual SRAM admission; the656-byte
margin is not permission to assume arbitrary scheduling code will fit. Full64-layer
correct sentence generation and2,000 dependent output tokens/s remain the target.
