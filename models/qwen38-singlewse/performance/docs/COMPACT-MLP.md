# Exact MLP scale sharing for recurrent integration

P44 admits the complete original layer00 fabric with smaller MLP banks. It
creates capacity for actual recurrent workers while preserving original FP8
codes, arithmetic order, K ownership, every existing route, all retained state
pages and both work slots. This is compiler/storage qualification, not neural
execution, multi-turn conversation or a model-speed result.

## Change and evidence

The original4x64 and8x32 native tiles each repeated one FP32 weight scale.
Consecutive local rows belonging to the same original128-row block can use an
identical scale word. The compact reader stores256 code bytes per tile and a
separate local scale table per branch and K part. No scale is shared across PEs,
different matrices or different original blocks; no value is requantized.

`CompactMlpPlacement` supplies the five-field native setup and matrix addresses.
`CompactMlpAuxiliaryPlacement` shifts retained page addresses only by the local
prefix saving. Original MLP auxiliary suffixes remain intact. The active bank
map is `mlp-bank-placement.json`; the retained P42 frontend map alone is no
longer a valid address resolver for these banks.

| Check | Result |
| --- | --- |
| Paired calibration032 |20 PEs; direct readers add48–96B, ragged8x32 readers save384B |
| Complete compile024 |11,388 PEs,8,428 ELF images, all admitted |
| Highest SRAM including4,096B stack |48,032B; smallest margin96B |
| MLP bank bytes removed |3,912,128B |
| Actual total SRAM reduction |3,362,352B after reader/code alignment |
| Original MLP scale words |1,044,480 ->66,448 exact local words |
| Independent bank reference002 |98,003,376 original32-bit words verified |
| Retained original mixer/state/work pages |28,605; exact values and PE identities preserved |
| Complete candidate bank bytes |388,101,376B |

The remote migration classifies every original word exactly once. Only equal
FP32 scale words may alias. It then reopens the packed banks and reconstructs
every weight/scale access independently from the actual CSL setup formulas,
including128-row block crossings and ragged K parts. Every original source word
matches on this fresh readback. Initial retained state contents are preserved;
this does not qualify runtime recurrent updates.

The full census independently checks every application coordinate and confirms
all old route/export declarations remain unchanged. All36 original non-layout
CSL files are retained byte-for-byte; the new compact native reader and its
caller are separate files. Changed numerical execution is still unqualified.

Twenty selected source tests and the same twenty site-adapted publication tests
pass, including original-block boundaries, ragged K ownership, all retained page
addresses, frozen route preservation and continuing-dialogue protocol regressions.
Code hashes are unchanged across each test run. This is not a full test-suite run.

## Preserved failures and resource limits

Complete compile023 reached its600-second deadline. Recorded memory adjustments
from2 to3 to4GiB responded to measured direct-reclaim pressure; no swap, CPU or
deadline limit was relaxed. The new bank sizes increased profile classes from
1,247 to1,481. This failed snapshot and both resource adjustments are retained.
Compile024 starts at4GiB with a900-second compiler deadline and completes in
520.129 seconds. Both services terminate and release the shared workstation
lock. Reference002 uses2GiB/one CPU and finishes in8.233 seconds. No new physical
WSE job is submitted for this milestone.

The earlier GDN math and rejected SDK-replacement calibrations028–031 are also
preserved. They do not change the selected complete graph. See
[recurrent integration](GDN-INTEGRATION.md) for their scope and the state/buffer
contracts needed next.

## Next boundary

Use compile024, reference002 and `compact-mlp-census-001` as the new complete-bank
base. The capacity sensitivity table assumes code/scratch costs and is not an
admitted GDN layout. Actual full-key state workers need an explicit bijective
retile, measured transport costs, compatible frontend returns and queue-enforced
buffer leases. Connect real recurrent computation to the qualified frontends,
output projection and retained norm/MLP graph. The complete64-layer continuing
conversation and>=2000 average output tokens/s target remain unmet.
