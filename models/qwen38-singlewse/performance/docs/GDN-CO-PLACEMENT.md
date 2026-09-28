# Complete-bank recurrent co-placement

The target remains one continuing original Qwen3.8-27B-FP8 conversation at
>=2000 average output tokens/s, including appended input processing and replies.
This work does not yet execute the recurrent graph or establish a model rate.

The private recurrent port receives token/decay/beta and complete V/K/Q vectors,
updates FP32 state, and emits scalar BF16 result frames. Existing native decoder
and operand arenas are leased across projection and recurrence phases. Local
return DMA completion permits source scratch reuse; it is not a consumer drain.
Explicit diagnostic reset/admit entries keep the actual port reachable without
resetting state on startup or supplying synthetic neural results.

## Why row ownership changes

Complete private ports cost about1.5–1.65KiB beyond the existing projection
program. The fixed P44 banks cannot hold every full-key state column at this
cost. Sharing the original MLP receive/send callbacks was measured in035 and
increased code size, so it is not selected.033's unsupported compile-time pointer
cast and034's two overflowing64-column samples remain preserved.

The new planner changes complete MLP output intervals within existing K cohorts.
Three gate/up cohorts and one down cohort take fewer rows. Other cohorts take
the displaced original rows. This reserves753 actual cohosts:748 workers with
eight value columns and five mixer cohosts with32. Together they retain all48
heads x128 keys x128 values, or786432 FP32 elements.

K cohort membership and tree shapes remain. Original rows can move between a
full cohort and the ragged tail; their floating-point reduction order can then
change. The original tensor bytes are unchanged, but the new neural result still
requires independent numerical qualification. Regenerated fused MLP routes cover
all17408 gate/up rows,136 quantization groups and5120 down outputs. The controller
retains one277-word response frame, following the new largest output sink.

## Complete storage identities

The old recurrent pages each hold4 keys x8 values. They are explicitly retiled
into full128-key arrays with worker-local contiguous value columns. All786432
logical coordinates have a checked inverse mapping. Source state must be copied
as values; assuming consecutive old pages form a column is invalid.

The old MLP auxiliary suffixes are also retained. They become explicit
`gate_up`, `down` and `mix` namespaces, rather than disappearing when matrix
prefixes change. There are7085 nonrecurrent pages, both work slots, all original
weight codes and original scales. The proposed banks occupy388094976 bytes;
changed output boundaries remove6400 additional duplicate scale-table bytes.
The initial plan moves898 nonrecurrent pages between PEs. The actual-census
refinement moves ten more; a fresh original-to-final comparison counts908
distinct changed-PE pages and5,226 changed addresses. These are storage quantities,
not speedups.

`gdn-bank-placement.json` is the candidate address authority. Its `stage` defines
the new matrix intervals, `workers` the recurrent state slices, and namespaced
`spans` every other auxiliary page. P44's source maps remain provenance only.
`GdnBankPlacement` resolves each namespace and state coordinate. Its
`recurrent_spans` returns one contiguous span for an8-column worker and four
strided row spans for a32-column worker; treating both as one old128B page is
explicitly rejected.
The planner reserves1632B per MLP cohost,1728B per mixer cohost and16B guard using
measured code costs. Only a complete actual ELF census can admit these estimates.

## Actual-census refinement

Complete compile026 produces all11,388 PE programs, but five32-column mixer
cohosts each exceed the48,128-byte application ceiling by112bytes. The refined
plan moves ten nonrecurrent pages and changes nine bank extents. All recurrent
state addresses, matrix prefixes, numerical code, native descriptors and routes
remain identical to026. This uses the complete linked census, including4KiB
stack, rather than extrapolating selected code probes. Fresh compile027 admits
all11,388 PEs/8,240 ELF images with maximum48,112bytes including4,096bytes
stack, and16bytes minimum margin. No PE exceeds the application ceiling.
The original-bank migration is gated by this complete result.

The compiler now defaults to16GiB, no swap and a1,800-second deadline, with
an8GiB host-headroom admission check. See
[Workstation compilation](WORKSTATION-COMPILATION.md).

## Original-word qualification

Remote reference003 verifies all97,025,344 original source words after flushing
and reopening the new banks. It covers all1,044,480 MLP tiles,7,085 nonrecurrent
pages,753 state slices and786,432 original FP32 state elements. Actual compiled
native descriptors independently address the readback. Banks occupy388,094,976B;
original scale sharing changes by6,400B without requantization. The reference
takes9.562s and releases its bounded workstation service. No model payload or ELF
is downloaded to the local workspace or committed to GitHub.

The source checks include45 selected regressions and seven targeted refinement
checks. The final site-adapted publication tree passes46 selected regressions.
All seven services release, including rejected/failed attempts. No new hardware
job is submitted. See `evidence/gdn-co-placement-summary-001.json`.

## Remaining execution boundary

The new private ports are callable, but frontend-to-GDN broadcast, scalar return
merging and the full downstream retirement fence are not connected in this
candidate. The P43 frontend expects paired BF16 returns; it cannot consume the
new scalar stream without a separately qualified adaptation or pair gather.

Before any neural acceptance, freeze numerical bounds for the reassociated
FP32 recurrence and changed MLP reductions, connect all48 actual heads, verify
persistent state across positions and new prompts, and test explicit reset/replay.
The selected8/32-column slices are even. Prefer a measured paired-return
variant that preserves the P43 native frame contract and halves return packets;
the current compiled candidate still emits scalar frames.

A whole layer must then execute its original output projection and norm/MLP graph.
All64 layers, full head and dependent token feedback remain required for serving
measurements. No source, address, compilation or bank-copy check substitutes for
that execution.
