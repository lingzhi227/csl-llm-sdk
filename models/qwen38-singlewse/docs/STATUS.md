# Execution status

## Latest: initial functional phase closed for publication

The user explicitly closed this phase after complete physical sentence generation
and the final small norm diagnostic, with communication optimization and gradual
speed improvements deferred to future work. No additional inference-code changes
or hardware experiments are being started.

Corrected `resident-generation-hw-003` completed37 generated IDs including EOS,
two complete sentences,64 processed positions through all64 layers and the full
vocabulary head. Its captured TTFT was838.693552495s; complete request1921.589062858s.
All1251 text tensors were loaded once; all870000 endpoint traces passed.
The complete compile004 and fresh role binding003 passed.

The final65-input norm diagnostic found3 corrected output differences from
PyTorch (332800 scalar values), at most1BF16 ULP, versus32 baseline differences.
The third layer's input norm at position0 is now bit-identical to PyTorch.
This local evidence does not certify all full-model numerical differences.

Strict CPU alignment did not pass. Reference002 was stopped at the user's phase
closure after2330 of4224 vector comparisons:35 full positions and20 layers of
position35. Maximum observed relativeL2 was22.7357% for layer outputs,22.8431%
for final norm and21.8294% for logits. Criteria and failure receipts remain intact.
Corrected reset/replay and capture-disabled sentence measurements did not finish.
The physical runtime was cancelled after the complete first capture, not a normal
four-request shutdown. All39 physical jobs are released; reference workers have
exited and the heavy-job lock is free.

See PHYSICAL-INFERENCE-RESULT.md and the functional-milestone-001 receipt for
publication scope. The historical chronology below retains its original scope.

The active objective is complete Qwen3.8 sentence generation on a single WSE-3,
including real end-to-end measurements. The user explicitly accepts dense
Qwen3.8-27B-FP8 and smaller genuine Qwen3.8 releases; parameter count is secondary.
The official catalog currently has no smaller release. Community 2B/4B/9B
distills use Qwen3.5 bases and have not been substituted.

## Implemented and verified

- Pinned official FP8 revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Audited all 66 safetensors headers and all 1,606 tensor extents/index entries.
  Full checkpoint payload: 30,866,663,264 bytes. Required text-network payload:
  29,468,003,328 bytes, including both full BF16 embedding and output matrices.
- Acquired all 66 shards and eight support files on Mass1: 74 files,
  30,879,968,808 bytes. Every file passed publisher SHA-256/Git-blob verification.
  The complete finite-value/header audit also passed: no FP8 NaNs; all scales finite and positive.
- Implemented FP8 E4M3FN weight decode and GEMV in CSL. Actual original
  layer-0 gate-projection tile is 272x128; 34,816 packed weight bytes plus
  three FP32 scales, with no full-matrix expansion.
- Simulator `fp8-tile-sim-004` passed all 254 finite encodings against a
  separate PyTorch decoder, all three numerical cases, persistent weight/scale
  readback and call counters, and normal shutdown. Maximum absolute error
  1.1417867540330917e-7 against the frozen FP64 dot-product oracle.
- Its actual ELF low section ends at 42,608 bytes. With the declared 4,096-byte
  stack reserve this is 46,704 bytes, below the 48,128-byte application ceiling.
  This audit does not include the future network communication program.
- Preserved failures 001 (reserved keyword), 002 and 003 (288-row SRAM limits).
  No limit or numerical tolerance was relaxed to accept them.

- FP8 dynamic group-128 quantization simulator `fp8-activation-sim-001`
  passed 12 groups covering finite encodings, midpoint ties, adjacent FP32
  values, zero and tiny inputs. All output bytes and FP32 scales matched the
  pinned vLLM convention/PyTorch conversion exactly. The combined `fp8-flow-hw-001` subsequently passed physical quantization and transfer checks for five groups. The complete 12-group activation-only suite remains simulator evidence.
- BF16 144x128 storage/kernel compile passed: 43,024 bytes plus 4,096 stack
  reserve = 47,120 bytes. Communication is not included in this audit.

## Storage placement and unqualified routing

The initial144-row BF16 storage assignment accounted for all text tensors with845,393 abstract
weight PE identifiers. A 2,048-token state budget leaves 14,297 application PEs
for additional control/communication. At 96 tokens it leaves 19,503 PEs.
The current 8,192-token placement budget does not fit. Collision-free physical storage coordinates now exist. Routes, all-role memory
and whole-model communication correctness remain unqualified.
The successful tile has only 1,424 bytes of remaining conservative SRAM budget.

## Physical execution and remaining work

Physical `fp8-tile-hw-001` and `fp8-flow-hw-001` passed. All four compile/runtime jobs completed successfully and released their physical systems. The flow combines device dynamic quantization, packed cross-PE communication, the phase-16 original weight tile and result transfer, across five dependent calls. Its heaviest physical ELF plus stack reserve is 46,496 bytes. The source staging and
bounded physical lifecycle reuse the GPT-OSS task's verified machinery by copy,
with provenance in REUSE.json. A shared hardware lock and active-job check
prevent collision with that task. The original projects are not modified.

The user reauthenticated the shared SSH connection. A new remote probe succeeds
on `cer-anl-net001-us-sr01`; the account had no active jobs before dispatch.
The earlier failure was the stale shared connection, not basic network reachability.

The independent CPU FP8 reference `reference-full-002` completed successfully: all 64 layers, original FP8 checkpoint, full 248,320 vocabulary, official template with thinking disabled, 19-token prompt and maximum 24 generated tokens. It generated token IDs [9419, 0, 248046], decoding to `Hello!<|im_end|>`, and stopped on EOS. This is a short greeting, with two dependent decode steps, not evidence of sustained physical sentence throughput. CPU generation is numerical reference evidence only.

Still required: actual executable resident placement, all-layer operator/fabric integration, full-vocabulary
head/selection, dependent-token generation, complete sentences, measurements and
physical numerical acceptance. The accepted GEMV uses real weights but test
vectors; it is neither a complete layer nor full-model inference.

Read ../STATE.json for current process identifiers and next actions, and
ACCEPTANCE.md for the completion boundary. Reconcile active services before
restarting acquisition or submitting another experiment.

Reference attempt 001 completed full prefill and generated `Hello`, then systemd-oomd killed it at decode layer 8. Attempt 002 drops consumed file cache and caches the 24.38 GB of original compressed language-layer tensors under a 26 GiB hard cap, no swap, shared heavy-job lock and 2 GiB machine-availability floor. Arithmetic and workload are unchanged. The large original BF16 embedding and head are still read in bounded tiles for this CPU-only reference.

Full matrix physical profile: 2,600 PEs, all 17,408 output rows and 5,120 input columns of original layer-0 gate projection. Preparation 001 failed before submission because the SDK environment lacks Torch. Its replacement NumPy oracle passed the 12-group/1,536-code PyTorch boundary fixture exactly before dispatch. Its accepted compile is 002 and accepted runtime is 003, as recorded below.

`configs/placement-96.json` assigns every original text tensor and short-context state to unique application coordinates. Mixed 40/48/136-wide reduction strips use 1,150 physical rows; 845,393 weight PEs plus 5,104 state PEs leave 19,503 PEs. These are collision-free coordinates, not an implemented fabric route or compiler admission. The bounded resident weight packer is source-only until checked against original fixture bytes and a compiled role map.

The bounded weight packer passed identity checks for all 2,560 FP8 gate tiles across all eight row phases and 40 BF16 tiles with 96 padding rows. This checks layout/weight byte preparation, not a whole-model physical upload.

Full-matrix 002 compiled successfully and all 2,600 application ELFs passed SRAM: maximum42,480+4,096=46,576 below48,128. The run client then exited with SIGSEGV (-11), before any numerical observations. Its hardware job was cancelled and released. Attempt003 reuses the exact artifact/CSL after source/hash checks, bounds each host weight copy to four rows (<16MiB of host words), and adds initialization/upload stage logs. Acceptance bounds are unchanged.

Longer CPU reference003 completed successfully: all64 layers and1251 text tensors,28 prompt tokens,39 generated tokens including EOS. It generated two continuous sunny-morning sentences. The bounded service exited0 and released the workstation heavy-job lock. All64 layer outputs and complete vocabulary logits remain on Mass1; compact receipts are in evidence/reference-full-003. This is CPU numerical reference evidence, not a hardware generation or throughput result.

Full-matrix003 completed successfully with unchanged CSL/artifact and bounded host copies. All three full projection calls passed the frozen FP64 bounds (maximum errors4.7364e-7,2.5054e-7,0); packet distribution, all2600 call counters and weight/scale retention passed. Runtime job wsjob-dyx4wghueraes7vbix2uvv succeeded and released normally. The prior -11 cause is not proven; the successful bounded-transfer run does not establish a network fault. No full-model inference or full-model performance claim follows from this operator result.

Recurrent-head001 preserved a compiler failure (dynamic fabric extent); its job is released. Recurrent-head002 then failed on pointer arithmetic and its job was released;003 uses element-address ptrcast. The subsequent003 qualified a full128-key x128-value FP32 head state using two value64 shards,96 updates and reset/replay, with controller broadcast and device output join. This synthetic preprocessed-operand test excludes convolution, normalization and gate nonlinearities.

Resident-bus001 is staged, not dispatched. Its45-PE prototype uses ten real FP8 tiles in four interleaved strips. A finite device controller sends all operand/reduction/collection commands; output travels east then south entirely on device. Every endpoint must acknowledge request completion before the host command returns. The candidate whole-model reservation keeps original weight coordinates and reserves1909 spine PEs, leaving17594 for remaining roles; it is not whole-model route qualification.

Recurrent-head003 passed physically:96 dependent state updates plus8 reset replay positions; all packets, counters, state/output error bounds and BF16 casts passed. Replay state/output matched the first run exactly. Maximum ELF low40,496+4,096 stack=44,592 below48,128. Compile wsjob-kjjzjdbhn2crkariqi37om and runtime wsjob-fosnmhtvl4zawv34che2a2 both SUCCEEDED and released. This is one full state head on synthetic preprocessed operands, not a complete GDN layer.

The complete semantic graph in configs/model-graph.json binds all1251 text tensors and scales to1172 operations across all64 layers, including400 FP8 projections and complete vocabulary. This is checked lowering input, not an executable full-model CSL artifact.

The 498-matrix controller descriptor table occupies7,968 bytes. All17,962 row-strip coordinates are reconstructed exactly and checked against the complete atlas; it is data for future device lowering. Candidate qwen_math/gdn_preprocess libraries are source only and have not passed compilation or numerical qualification.

Bus002/003 both hit the SDK AstToCsl PointerType assertion, before runtime. The buffered-relay change alone did not resolve it. Bus004 removes runtime conditional expressions and passed compilation and physical execution. Both jobs SUCCEEDED and released. All45 PEs passed command/forwarding counters; three real-weight cases passed frozen numerical bounds (maximum error3.336e-8), with exact retained weights/scales. Maximum ELF low43,088+4,096=47,184 below48,128. The staged identical-CSL workstation simulation is superseded and will not be run. This small interleaved-strip pass does not qualify whole-wafer timing, fan-out or routing.

Task-local bounded SDK artifact upload passed local read-bound/mutation tests and real installed SDK protobuf comparisons with a recording stub (no network jobs). Future staged runtimes use this adapter; shared SDK installation and old accepted snapshots remain unchanged. Complete-model upload is still untested.

Qwen-math-hw-001 was rejected at shared-lock admission before any job submission. The exact CSL/fixture is being qualified in bounded workstation qwen-math-sim-001 while the lock is occupied. A new BF16 resident prototype adds original embedding first/last vocabulary tiles, BF16 head strips, variable-length return and masked BF16 argmax; it is source only until compiled and checked.

Qwen-math-hw-001 subsequently passed unchanged CSL/fixture on physical CS-3 after fresh admission. All three original-gain RMSNorm/in-place cases and128 nonlinear probes passed, weights retained, and both jobs SUCCEEDED/released. The task-local bounded upload adapter was exercised successfully for this artifact. This does not qualify multi-GB whole-wafer upload. BF16 resident001 compiled locally; SRAM and numerical checks are pending.

BF16 resident144-row simulation001 compiled but exceeded the conservative SRAM gate by320B; no numerical run was started. New140-row simulation002 passed SRAM(max43,312+4,096=47,408) and three numerical/embedding/argmax/route cases, then hit its300s cap after observations, during retention/shutdown. It is not accepted simulator completion. The identical140-row CSL/fixture subsequently passed full physical resident-bf16-hw-001, including original weight retention and normal stop; both jobs SUCCEEDED/released. The revised complete atlas uses849,313 weight PEs and leaves13,674 after96-token state and1909 bus PEs. Full graph code/routing is still incomplete.

GDN head001 failed fixture preparation before dispatch on NumPy scalar dtype promotion.002 fixed that but its pre-device propagated state bounds were too broad to be meaningful, so it was never dispatched.003 changes synthetic a operands to exercise moderate decays through original learned parameters, with independent oracle quality gates: maximum state bound0.000124, output bound8.34e-6, gated BF16 interval width0.01014. It is staged and admission requested; original convolution/gates/norm weights,96 positions and8 reset replay positions remain fixed before candidate observations.

GDN head003 passed physically: original layer0 head0 convolution/gate/norm parameters,96 dependent positions,8 reset replay positions. Packet intervals, all FP32 state/output intervals, gated BF16 intervals, exact convolution histories, counters, parameter retention and bit-identical replay all passed. Both jobs SUCCEEDED and released. This connected head consumes synthetic projected BF16 operands; full projection/GDN/all-layer integration remains unfinished.

Attention-group001 now has a frozen independent FP64 interval fixture: original layer3 Q/K norm gains, pinned Torch FP32 rotary frequencies,96-position capacity and all6 query heads for one KV head. Five-PE code is staged for local compile; no attention physical job has been submitted. Complete GDN/attention/MLP graph integration remains unfinished.

Attention-group-hw-001 passed physically after shared-lock admission retry:96 positions,all6 queries in one full KV group,4-position reset replay; QK norm/partial64 RoPE,KV retention,causal softmax,sigmoid gate,zero cache tail after reset and original parameter retention all passed. Both jobs SUCCEEDED/released. Source uses original layer3 norm gains and synthetic projected BF16 inputs; it is not a full layer/model.

The complete candidate role plan now assigns adjacent GDN/attention groups and16 full-sized activation banks without moving original140-row-atlas weights. It also reserves3183 candidate-output-only validation banks for all layer outputs,full logits and final norm up to96 positions, leaving8347 PEs. The complete lowered883-instruction program binds all1251 text tensors and uses15612 bytes of controller tables. Its device interpreter and combined role wrappers remain to be implemented.

Bounded ALCF acquisition001 is active separately from hardware jobs:512MiB address-space cap,5400s wall cap,16MiB/s download cap,task-owned model directory and original publisher hash checks. It does not allocate a wafer.

Bank-chain-hw-001 passed physically with the unchanged normal CSL and frozen fixture: original gate544x256 -> resident BF16 bank -> SiLU/BF16/dynamicFP8 -> original down544x384. All three numerical/bank/route cases, full weight/scale/bank-tail retention and normal stop passed; both jobs SUCCEEDED/released. No intermediate hidden state entered from the host. The simulator diagnostic timeout cause remains unproven. This is a connected submatrix chain, not full MLP/model inference.

ALCF SSH remains healthy. Acquisition001 could not reach publisher HTTPS and was stopped by proven PID identity; no payload began. Transfer001 failed before receiver payload on an AS ceiling below measured Python baseline;002 uses512MiB AS,1MiB streaming buffers,16MiB/s cap and bounded workstation service/shared lock. Publisher-verified model bytes relay over existing SSH pipes without Mac payload files.

Candidate complete interpreter and shared role wrappers are now implemented in resident/*.cslpart and generated/*.csl. All883 instructions,48 GDN/16 attention layers, original full embedding/head, resident intermediates, capture banks and device token feedback are represented in executable CSL source. This has not run a full model. Representative144-PE compilation005 passed:66 ELF programs cover all144PEs exactly once, maximum43728+4096=47824 below48128. Earlier export syntax failures and80B BF16 SRAM overage are preserved.

ELF rectangle decoding was checked against the installed SDK ELFObject for every segment in all66 compiled programs (audit004). A streaming artifact auditor now checks actual rectangle coverage instead of equating program files with PE count, verifies embedded source identities and every application SRAM gate, and retains raw records remotely.

Whole-layout compile001 failed because COMPILE upload kept only the final1MiB message of a4.3MiB source. Its compiler diagnostic first line exactly matched byte4194304. Compile sources now each use one bounded message<=8MiB. Compiler transport qualification003 passed real installed protobuf framing equivalence and malformed/multi-chunk download checks without physical jobs. Whole-layout002 failed on a comptime pointer;003 uses value-returning coordinate functions and is active. No whole-model physical runtime has been allocated.

Full acceptance criteria and prompt IDs are frozen in configs/full-acceptance-v1.json before candidate full outputs. The new resident_loader.py and resident_run.py are candidate host code; loader qualification, offline sequential-prefix CPU comparison, complete physical numerical acceptance and measured sentence generation remain outstanding. Check STATE.json for current compile/transfer ownership.

Whole-layout003 compiled successfully and released. Actual434 ELF programs cover870000 application PEs exactly once; maximum43728+4096=47824<48128. Compressed artifact46629402B, SHA185c9bf26bf4454256a43efb63d7278143b0707193d8750df1980bef7f3bd0d2. Role binding002 independently checks source roles and live weight/state/table object extents over every coordinate. SDK single-message upload qualification002 passed with oversized files rejected before transport. Full physical runtime001 is staged, not submitted; model transfer and new small-parameter read audit remain before admission. Offline sequential-prefix evaluator source is implemented but not run against a physical candidate.

All74 checkpoint files (30879968808B) reached ALCF with publisher verification. Parameter audit001 exposed a host traversal issue:400 scale aliases must not be uploaded as standalone allocations. Runtime snapshot001 was never dispatched. Snapshot002 skips those aliases, requires exact1251-name coverage, and passed parameter audit002. Complete runtime wsjob-h6ll2mmhvzdsnbkim7tayd is admitted under the shared lock; source/artifact unchanged from full compile003. No complete physical outputs or full-model performance result exists yet.
