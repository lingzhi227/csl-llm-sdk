# Performance milestones

## P0: checked graph and executable stream foundation

Status: source and simulator milestone. The full-model 2,000 tokens/s target has
not been reached. Physical transport qualification completed separately in P1 below.

Implemented and checked:

- All 1,172 original semantic operators, 1,251 text tensors and 64 layers imported into
  an explicit dependency/region representation; intra-token KV ordering and
 113 delayed state/token-feedback edges are explicit. Physical region placement
  is unassigned, rather than filled with hypothetical coordinates.
- Deterministic stream-to-PE-to-CSL lowering for a rooted 2D multicast and
  acknowledgement tree; manual geometry, packet, color, queue and task controls.
  The restricted verifier checks route continuity, resource collisions, coverage
  and estimated SRAM. This is not a generic arbitrary-graph deadlock proof.
- Seven source tests pass. Complete 3x4 simulator suites 006/008 pass all 5 cases,
  including changed inputs, restarted requests,33/256-word payloads and 128 rounds.
  All endpoint packets, epoch counts, child counts and independent root sums pass.
- 16x16 simulator 007 compiled all 256 PEs and passed its first 2 cases, including 64 rounds
  of 33 words at 2,176.03125 root cycles/roundtrip. It was stopped when observed simulator
  cost showed the complete suite would exceed its 240-second budget. This is a
  partial simulator result, not a full physical or full-model performance result.
- An independent IEEE encoding proof covers 254 finite FP8 codes and 64,516 products.
  It identifies a possible table-free mixed-precision kernel, with device
  subnormal behavior and accumulation correctness still unqualified.

Failed attempts are preserved. Simulator 001 rejected non-symbol export operands;
002 hit the bounded process/thread limit before actual compilation; 003 timed out;
004 exposed a device assertion, and 005 captured the actual checker error: epoch 1
returned the correct sum 90 while its multiply-based check evaluated to 65,610.
006 uses an incrementally maintained device check while the host independently
retains the closed-form formula. No numerical tolerance was relaxed. The cause
of the original integer expression requires a separate compiler/arithmetic probe;
it is not presented as a proven general compiler defect.

For the initial local compile, explicit process CPU affinity corrected LLVM's
thread creation failure without raising the 64-task limit. Simulator fatal-log
monitoring now terminates a halted simulator promptly. Hard memory/no-swap,
CPU/runtime and shared-lock limits remain in force.

## P1: physical 256 PE stream qualification

`mesh-transport-hw-001` completed all five physical cases on one WSE-3. Every
endpoint packet, epoch count, child acknowledgement count and independent root
checksum passed, with changed inputs and restarted requests. The 33-word cases
measured 2,248 root cycles per roundtrip, and the 256-word cases measured 2,500.
These are complete multicast plus all-endpoint acknowledgement cycles, not model
token latency. Clock frequency is not assumed. Host completion plus audit transfer
is reported separately in the raw compact results.

Both jobs succeeded and released normally, with no cleanup errors:
`wsjob-hd2dkp3emcbrcfmv9ytx2d` (compile) and
`wsjob-nwki8fjpbhdyphj3rs2kfw` (execute). Stage wall times were 232.454 seconds and
71.407 seconds, including infrastructure work. They are not provider-billed node
hours. A fresh account/system audit found no owned active jobs or assignments.

All 256 compiled application ELFs passed the local SRAM gate; maximum low-section
end plus the declared 4,096-byte stack allowance is 10,752 bytes, below 48,128.

## P2: exact native mixed-precision products on WSE-3

`fp8-native-hw-001` passed all 64,516 ordered products of finite FP8 encodings,
including encodings that become FP16 subnormals, plus two 128-repeat accumulation
cases and weight retention. Actual FP32 outputs equal the independent expected
finite values exactly, with zero signs canonicalized by the positive-zero FMA
accumulator. Original tensor weights, packed extraction and complete GEMV are not
yet part of this probe.

A 254-element native mixed FP16/FP32 FMA vector takes 135.3125 to 135.3203125 cycles
per iteration in this loop. This is an operator measurement, not a model token
rate. These results motivate smaller per-PE matrix tiles and interleaved weight
banks, rather than assuming a large resident tile can meet the per-stage budget.

Compile `wsjob-6knikeasu8pkw2bv6p3gzg` and execute
`wsjob-btccu78jtevwtjzupy3y5h` both succeeded and released normally. Their stage wall
times were 41.464 seconds and 71.394 seconds. A fresh resource audit confirms no
owned active hardware jobs or system assignments.

The compiled one-PE program uses 10,128 bytes including the declared
4,096-byte stack allowance, below the 48,128-byte application ceiling.

## P3: physical packed FP8 original-weight tiles and bank ownership plan

`fp8-tile-fast-hw-001` passes all 65,536 packed two-byte patterns through the
vector bit insertion transform, and 32 dot cases built from eight original
layer-0 matrix tiles and four fixed activation patterns. NaN encodings participate
only in the bit-transform test; numerical fixtures contain finite original FP8.
The original shard and fixture are hash pinned. Both PEs receive identical inputs:
one executes unchanged scalar FP8 arithmetic, the other vector decoding followed
by native mixed FP16/FP32 arithmetic in the original 128-term order.

All optimized outputs are bit-identical to the scalar outputs (zero signs
canonicalized), finite, and within the frozen independent FP64 absolute-product
bound. The maximum observed FP64 error is 3.814697265625e-6. All weights, scales,
packets and call counts pass retention checks; normal stop completes.

Physical 2x128 tile totals are **11,316 scalar cycles versus 1,286 vector/native
cycles**, an 8.799x local speedup. The latter includes 829 cycles decoding weights
and activations, 424 GEMV cycles and 33 finalization cycles. These boundaries include
timestamp overhead; there is no assumed clock frequency. Standalone extraction
of 2,048 bytes takes 3,070–3,071 vector cycles versus 30,734 scalar cycles.
This is not a complete projection, inter-PE reduction, overlap or model token rate.

Both jobs succeeded and released normally without cleanup errors: compile
`wsjob-hwtzlyzcypfouevqvpgjxb` (41.416 seconds stage wall) and runtime
`wsjob-jgkalcjj5cwrnfhdbym5u2` (71.417 seconds stage wall). Maximum compiled SRAM
plus the unchanged 4,096-byte stack allowance is 18,208 bytes. A fresh account and
system audit confirms no owned active job or assignment. Simulator failure 001
(reserved parameter name) and complete simulator 002 remain frozen.

The separate compact weight-bank planner binds all 1,251 original tensors,
including 498 matrices and 400 FP8 scale aliases. Under the explicit rows=2 policy,
853,616 PEs hold compressed matrix banks and 16,384 PEs are reserved for other
actors. The bank estimate includes replicated FP32 scales, four descriptor bytes
per tile, 6,144 code bytes, 4,096 stack bytes and 1,536 scratch bytes. Its maximum is
47,536 bytes under the 48,128-byte ceiling, leaving only 592 bytes. **This is an
estimated ownership/storage result, not compiled whole-model SRAM admission.**
Actual routes, descriptor encoding, code size, non-matrix state and complete
execution remain to be implemented. Ten source tests pass, including independent
cyclic-ownership enumeration and cross-matrix local-offset uniqueness.

## Next evidence

Compile a resident bank actor with actual buffer/code extents; expose tile
predecode and operand receive as separate readiness events. Then measure regional
GEMV, numerically qualified reduction and overlap before complete-model integration.
No component milestone satisfies the full-model 2,000 tokens/s target.
