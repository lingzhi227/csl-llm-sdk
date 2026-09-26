# Execution state

Updated 2026-09-26 UTC. The authorized target is complete original GPT-OSS-20B
inference in CSL on one WSE-3: west token input, resident original parameters,
on-wafer neural computation, east token output. **Complete-model two-token
physical inference passed numerical qualification on 2026-09-26 at 06:33 UTC.**
This is a bounded functional acceptance, not a long-context or corpus evaluation.

## Accepted complete-model attempt

`full-model-hw-003` is complete: original tokens 13225 -> 11 -> 5922, all PE
protocol checks, exhaustive unchanged weights, normal stop and released jobs.
Compile `wsjob-pyuzimpkuxchzpwjpf2kb2`; run `wsjob-gwfpx94zw3bsivg6svrzw5`.
All 2009 actual programs pass SRAM admission (38,960 + 4,096 = 43,056 bytes
maximum, ceiling 48,128). Host-observed forward times were 0.88284 / 0.89573 s,
excluding initialization, diagnostics and readback. These are not throughput
benchmark results.

The predeclared independent actual-input qualification passed all 48 layer-step
cases and 192 selected-expert invocations. RMSNorm is within 1 BF16 ULP of FP64;
all 552,960 SwiGLU outputs exactly match original Torch on the same input;
all 1,658,880 expert projection outputs satisfy their FP32 accumulation bounds.
No router or join interval violations. Maximum local attention relative-update
L2 error is 0.00013343; original head on actual final states selects both correct
tokens. See `evidence/full-model-hw-003/COMPLETE.json`, `QUALIFICATION.json`
and `qualification-summary.json`.

The strict global CPU-reference check remains **false**, unchanged: last-layer
relative L2 differences are 0.009225 / 0.012884. Exact tied top4 scores and
BF16 rounding propagate between layers. Acceptance uses the predeclared
`docs/NUMERICAL-QUALIFICATION.md` operator criteria plus exact generated tokens,
not a relaxed global tolerance. The two-token scope is explicit in the receipt.
All owned physical and CPU jobs are inactive/released. No active attempt remains.

## Preserved earlier complete-model attempts

`full-model-hw-002` ran all 24 layers and the full head on physical WSE-3. All
459 tensors were loaded once (73,893 transfers). The east endpoint returned 11
for input 13225, and all PE epoch/handoff checks passed. The original strict
per-layer comparison then failed and the job stopped/released before step two
and exhaustive retention. It is not a completed full-model acceptance.

Offline original-operator replay on the actual captured inputs found 15/24
attention and 14/24 MLP outputs bit-exact; largest local relative L2 difference
was 0.00054549. Layer 10 expert IDs 7 and 30 have equal BF16 router scores;
CSL deterministically chooses the lower ID while Torch topk chooses 30.
The complete layer-23 accumulated relative L2 difference was 0.00850.
See `evidence/full-model-hw-002/operator-replay.json`.

`norm-accuracy-hw-001` subsequently passed all 144 original/captured-input
RMSNorm cases on physical hardware. Compensated reduction reduced differences
from direct FP64 BF16 from 109 to 4 among 414,720 values, all <=1 BF16 ULP;
only two differ from Torch. Both jobs succeeded/released. The FP64 oracle
quantizes directly to BF16, avoiding one earlier double-rounding error.
The partial simulator run and earlier oracle discrepancy remain preserved.

The first full compile succeeded: 870,000 application PEs, 2009 shared programs.
All 2009 ELF SRAM checks passed in attempt 002, max ordinary section end 38,944
plus declared 4096-byte stack allowance = 43,040 bytes. The archive is
67,881,054 compressed / 1,223,945,197 unpacked bytes. Attempt 001's old 512 MiB
unpacked-size rejection is retained; 002 revalidated the same artifact under
a declared 2 GiB unpacked bound without recompiling. The norm change in 003
requires a new artifact, rather than reusing those earlier binaries.

The completed full inference used an 1800-second deadline, 8 GiB client
address-space limit and 2 GiB sampled RSS limit. Independent CPU qualification
ran for 96.7 seconds with two threads and a 4 GiB service memory bound.
Strict original comparisons remain preserved in the physical receipts.

## Checkpoint and reference

- The complete publisher-verified checkpoint is on both workstation Mass1 and
  ALCF: revision `6cee5e81ee83917806bbde320786a8fb61efebee`, 459 tensors,
  13,761,264,768 original payload bytes. All three shards and ten support files
  passed SHA-256 verification at both destinations. See `evidence/acquisition/`.
- Original MXFP4 blocks/scales and BF16 other weights are retained, without
  expert pruning, layer truncation or persistent FP16 expert expansion.
- Original OpenAI Torch forwards at commit
  `7b583341fe16729127f6d5b94a7b09ccae97e1a1`, with loading adapted to the HF
  checkpoint, produced `Hello` (13225) -> `,` (11) -> ` World` (5922).
  Per-layer reference states and routing are saved remotely. This CPU result
  is an oracle, not a CSL inference result.
- Complete placement/packing audit passed for all original tensors: 73,893
  bounded transfers, 27,556,745,472 host-slot bytes. It verifies loader coverage,
  not full-model device residency. See `evidence/resident-loader-audit-001.json`.

## Accepted physical evidence

All accepted experiments below stopped normally, verified unchanged resident
weights exhaustively, and confirmed every owned job released.

| Evidence directory | Scope and result |
|---|---|
| `mxfp4-tile-hw-001` | Original 160x288 expert tile, three cases, maximum absolute error 5.29e-7 against FP64 dot oracle |
| `mxfp4-flow-hw-001` | West input -> packed GEMV -> east output, three epochs |
| `moe-math-hw-001` | Top4, selected softmax, clamped interleaved SwiGLU and weighted join, BF16 exact |
| `transformer-math-hw-001` | RMSNorm, YaRN, learned attention sink and two persistent-KV epochs, checked outputs BF16 exact |
| `mxfp4-chain-hw-001` | Full 2880-contraction expert-row chain, all three cases passed |
| `expert-full-hw-001` | One complete original expert 5, 828 PEs, all 23,040 checked BF16 values exact over two inputs |
| `router-hw-001` | Original RMSNorm and 32x2880 router, two exact selections [5,18,9,16] and [23,10,28,22] |
| `moe-layer-hw-001` | All 32 original experts resident, top4 join/residual, 31,320 PEs, all 5,760 final BF16 outputs exact |
| `attention-prefix-hw-001` | Complete original attention, 4,640 PEs, both 2880-value outputs exact; K cache exact, one V-cache value differs by 1 BF16 ULP |
| `decoder-layer-hw-002` | Complete original layer 0, attention directly feeds MoE on wafer, 31,320 PEs; both attention and final outputs exact (11,520 values total), persistent KV, exact top4 |

The decoder loaded all 19 original layer tensors (476,863,552 bytes) once.
It reused unchanged CSL and the successful compile from `decoder-layer-hw-001`
(`wsjob-dwespfadhkaawecsds7mvc`); accepted run
`wsjob-dufco5udarbcncn36tjam5`. The earlier run
`wsjob-yex3kdvkpmppq5dbk4vaqi` failed before neural computation because reading
one embedding row mapped the entire 1.158 GB tensor against a 4 GiB client
address-space limit. Bounded row mapping fixed the host driver; the failed job
was released and its evidence retained.

## Implemented complete backend

- `core/resident_geometry.py` and `tools/build_full_layout.py`: 750x1160
  application, 24 layers each 27 columns, complete embedding and untied head.
- `core/resident_loader.py`: every original weight, scale, bias, norm and sink
  has an owner; initialize once, retain across tokens.
- `csl/resident/`: BF16 projections, packed MXFP4 experts, all original MoE and
  attention semantics, explicit collectors and layer handoffs, full-vocabulary
  greedy selection at the east endpoint.
- `core/resident_schedule.py`: 1990 fixed phase/layer/index commands per token.
  After initialization the only neural H2D input is the west token ID. Host
  commands do not contain intermediate activations or expert choices.
- Initial runtime capacity is 96 tokens. It retains all required causal history
  for positions 0..95, including the original window-128 layers, and rejects
  overflow. This is not evidence for 8K or the model's maximum context.
- Full driver feeds the actually generated token back for the second step.
  It reads all layer diagnostics only after the complete token, never forwarding
  them into computation. Strict original-reference checks are retained. Actual-
  input operator qualification additionally handles legitimate BF16 accumulation
  and equal-score top4 differences; exact greedy IDs, all PE epochs, exhaustive
  retained-weight readback, normal stop and released jobs remain mandatory.
- Current acceptance harness uses the fixed two-token `Hello` reference.
  Arbitrary-prompt generation and long-context validation remain outstanding.

## Compilation and preserved limitations

- Whole decoder: 266 shared programs; largest ordinary section end 38,368
  plus declared 4096-byte stack allowance = 42,464 bytes.
- `resident-roles-compile-002`: 180 representative edge programs pass actual
  ELF SRAM gates. This is compile-only evidence, not a runnable full graph.
- Local full compile 001 was stopped to reduce CSL interpreter work; 002 was
  stopped after finding a cross-phase handoff hazard; 003 was stopped before
  completion at 3,080,597,504 bytes of its 3 GiB resource bound. An unrelated
  Qwen CPU reference occupied most workstation RAM at that time (now finished). None is a successful full
  compile. The physical compilation succeeded with an 8 GiB SDK resource request;
  the cluster applies its own coordinator resource floor.
- Producer and receiver now participate in the same command: attention -> MoE
  in phase 9, layer L -> L+1 in phase 14(L), final layer -> head in phase 14(23).
  Waiting to receive in a later command can prevent the current command from
  draining. All frozen full-model sources include this correction.
- Mini expert simulator 002 passed both numerical epochs, then timed out during
  exhaustive readback. No simulator normal-stop claim; complete physical
  expert acceptance subsequently passed. Earlier chain simulator timeouts were
  preserved; chain simulator 004 completed under its declared 480-second bound.

## Storage and ownership

Sources and compact evidence are local. Model payloads, NPZ arrays, ELF files
and SDK archives remain on remote storage. Synced `sources/` and Qwen work are
read-only references. No unrelated job is cancelled or changed. Historical
`configs/resident-plan.json` is an earlier storage-capacity proposal; current
executable coordinates come from `core/resident_geometry.py`.
