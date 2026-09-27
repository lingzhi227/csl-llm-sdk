# Design antecedent: execute BF16 banks beside the qualified FP8 path

P10 now implements and physically qualifies the representative combined bank;
see [MIXED-BANK.md](MIXED-BANK.md). The proposal below is retained as design history,
not the current implementation status. Complete-model scheduling remains unfinished.

The original untied embedding/head and other BF16 matrices remain required.
P7 reserves their bytes but does not execute them. Adding another isolated FP8
benchmark cannot close this gap; the next bank component must consume real
BF16 slots with actual code, temporary storage and descriptor leases admitted.

The installed compiler supports a single runtime16-bit floating format per
program, selected by `--fp16-format` (confirmed by its driver help and the
[CSL type-system documentation](https://cerebras-sdk-docs-140.netlify.app/csl/language/types)).
The current FP8 native path relies on IEEE half's exact E4M3/256 embedding.
Changing the entire program to BF16 would require requalifying that path and its
packet representation. Retain the qualified format while adding lossless BF16
high-half expansion into FP32 for the other matrices.

The preserved functional `csl/bf16_gemv.csl` and `csl/bf16_resident.csl` already
implement this arithmetic: expand only the current column's BF16 weight words
into FP32 scratch, then perform ordered native FP32 FMA. A small-row adaptation
can keep a2-element scratch array and stream weights rather than expanding a
complete512-byte tile into1024 bytes. Using `@map` over packed BF16 activation
words may replace the original scalar indexing loop. Each activation expands
exactly by placing its16 bits in the upper half of an FP32 word.

The existing expansion uses DSR3, which conflicts with some pending dataflow
receives. A new module should own an explicit independent lease (candidate DSR7)
for streaming weight expansion, alongside DSR4 for the dot. This lease and code
must be composed with the actual resident-bank/network program, not admitted by
adding independent footprint estimates. P7's48-byte margin is insufficient for
an assumed additional path. Keep original capacities unchanged until a compiled
combined result or a full-model conservation-checked redistribution justifies a
new policy.

Qualification must pin original embedding/head and other BF16 row tiles, cover
signed zeros/tiny values, exact high-half expansion, repeated FP8/BF16 switches,
first/middle/last slots and complete bank retention. Check original ordered FP32
results and independent FP64 error bounds; measure the full local operation and
actual SRAM. Preserve the functional modules and record copy/adaptation hashes.
This note is a next-step design, not an implemented or measured fast BF16 bank.

A concrete next composition can build on P9's smaller contraction executable
instead of the P7 return/ack protocol. P9's maximum12,432-byte footprint already
includes one260-byte FP8 tile/scale, packet/decoder buffers and4KiB stack. Replacing
that one tile by112 resident FP8 tiles and adding12 real BF16 tiles gives the
arithmetic estimate12,432+111*260+12*512=47,436 bytes before added BF16 code and
slot-selection state. This leaves692 bytes against the unchanged48,128 ceiling.
It is only a planning estimate; compile the complete candidate to admit it.

This candidate need not retain P7's unused per-tile descriptor-sized sentinels if
it implements an explicit streamed slot/type dispatch contract. The controller
must provide checked slot/type ownership, and eventual full-model lowering must
conserve every original tile and compute those addresses. Eliminating an unused
capacity array does not establish the missing full model schedule. Record the
changed metadata policy and controller costs rather than pretending the original
four-byte-per-tile descriptor estimate still applies.
