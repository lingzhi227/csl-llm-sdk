# P41: fused projection output and shared-Q/K frontend admission

This milestone implements a direct device packet path from complete original
QKV/Z/A/B roots to16 frontend groups. It does not execute a complete neural
stage, recurrent core, conversation, or throughput benchmark. The unchanged
whole-layer bank placement is explicitly rejected by the new resource planner.

Each frontend owns one128-element Q head, one128-element K head and three
128-element V heads. Its source implements the original four-tap convolution,
BF16/SiLU boundaries, Q/K normalization, gate calculation, recurrent-input
packet leases, and post-core gated RMS normalization. Three BF16 history samples
per original channel persist across token invocations and turns. Q/K history is
shared instead of duplicated across the three V heads. An explicit drained reset
clears this history. These newly composed numerical operations still require
independent numerical execution qualification. Their parameter caches are
intended to receive exact original checkpoint slices; this milestone does not
load or independently verify those cache values.

The current compiled root variant writes the tagged frontend packet directly
into the projection's already leased output buffer. It reuses the existing
projection sender DSRs, microthread and completion task. The separate copied
packet and source callback used in earlier preserved attempts are removed.
Source DMA completion permits reuse of that output buffer; it is not downstream
retirement. Device input packets retain a32-bit token invocation, semantic row,
and two BF16 values. Invocation is the shared token request ID, not the distinct
projection generation number. Final gated-output retirement is separate from
recurrent-input packet retirement.

The candidate network contains88 source roots,1216 bounded merge actors and16
consumer groups. Software merges alternate vertical colors3/9 with whole-packet
backpressure and round-robin input selection. Horizontal colors0/1/19 use tagged
row broadcasts. Norm sender/bridge rows143/144 are router-only transit for this
network. Static tests walk all1408 root-to-group paths, cover all16480 original
projection values exactly once, and reject dropped/aliased/misfiltered paths.
The complete original emitted router configuration has no PE/color collision
with these additional routes. Total input traffic is8288 five-word packets;
this count is not a speed measurement.

## Compiler observations and rejected placement

| Attempt | Observed result |
| --- | --- |
| backend018 | Rejected: `color` shadows a CSL builtin type. |
| backend019 | Rejected: local task29 is reserved by the existing teardown handler. Frontend notifications move to free tasks18/19 on standby cohosts. |
| backend020 | Code compiles; full-bank consumer SRAM exceeds the48128-byte gate, reaching48672 including4096 stack. |
| backend021 | Preserve V in its already rounded BF16 format, expanding only at recurrent packet preparation. All33 selected programs pass; all16 consumer groups keep their complete original bank extents. |
| backend022 | Replace bounded head-index division with an exhaustively checked integer expression; selected programs pass. |
| backend023 | Fuse packet encoding and sending into the original projection output lease. All33 selected programs pass, with maximum consumer47984bytes including4096 stack and minimum144bytes margin. |

Most non-consumer samples deliberately use4096-word calibration banks. They are
not full-bank admissions. The direct fused root sample adds an estimated388bytes
relative to its P40 cohost, compared with932bytes in021. Selected ELF sizes are
observations; expanding reduced banks and applying a maximum cost to unsampled
cohosts remains a planning estimate.

`frontend-composition-audit-002` binds the actual selected ELF census to the
complete P40 per-PE census. With an explicit128-byte unsampled alignment reserve,
the current layout has1275 estimated conflicts with immutable matrix prefixes,
and its extra-code reservation is1909276bytes. Moving only auxiliary/state pages
cannot admit this candidate. The planner therefore forbids dispatch of this
unchanged whole-stage layout. Original functional and P40 source/banks remain
preserved; there is no new whole-stage compiler or physical acceptance claim.

The next lowering must jointly account for original matrix ownership and
communication/consumer code. A useful candidate is contiguous Z-head ownership
(to reduce root fanout and the number of software merge positions), followed by
full-K group rebalance using measured cohost budgets. Any changed tile map needs
an exact migration of original bytes, complete descriptor coverage, a fresh
whole-stage ELF census, and numerical qualification. Recurrent state workers,
actual recurrent return packets, output-projection ingress and the complete
layer event schedule are still unconnected. Do not replace this work with more
standalone speed sweeps or host intermediate gathering.

All27 selected source tests and27 publication tests pass; this is not a full
regression-suite run or numerical frontend execution. All six bounded
workstation compiler services are released. No hardware job was
submitted in P41; the closing account snapshot has no owned active job or system
assignment. Provider billing is unavailable in that snapshot. The single-dialogue
2000tokens/s acceptance contract remains active and unmet.

Evidence: `evidence/layer-backend-compile-018` through `023`,
`evidence/frontend-composition-audit-001` and `002`,
`evidence/frontend-tests-001.json`, `evidence/frontend-publication-tests-001.json`, and `evidence/resource-audit-p41-close.json`.
