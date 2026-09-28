# Next work: resident spatial pipeline

Immediate P41 follow-up: selected fused roots and all16 shared-Q/K frontend
consumers compile, but the unchanged P40 weight layout is rejected by
frontend-composition-audit-002 (1275 estimated matrix-prefix conflicts). Moving
state pages alone is insufficient. Jointly lower matrix/head ownership and the
packet network, consider contiguous Z ownership to reduce fanout/merge sites,
then requalify original-byte migration and every complete-program SRAM gate.
Do not dispatch the rejected layout or present selected calibration banks as
whole-stage admission. Qualify the new frontend numerics and connect actual
GDN state/return traffic and output-projection ingress before layer acceptance.
See [FRONTEND-FUSION.md](FRONTEND-FUSION.md).

1. Preserve physical006 as the accepted shared-input complete-MLP baseline:
   all original numerical/retention/drain checks pass,49376->24688 broadcast words,
   nonzero counts564289–564333. Four host observations improve versus P29 but
   remain slower than P28; no sustained wall speedup is established. Physical
   maximum48128incl4096stack controls admission, despite local compile48112.
2. Preserve accepted physical008 as the residual/RMS+MLP+residual/next-RMS
   integration baseline:81920 exact BF16 values, all native operands/counters,
   resident bank/gain retention, warm replay and normal stop.007's timeout and
   obsolete output-grant IDs remain preserved;148 source/publication tests pass.
   The state-address resolver now checks all49152 original pages through the800
   relocations and must be used by mixer lowering. Keep this integrated graph;
   advance to complete mixer/layer execution rather than an open-ended isolated
   optimization sweep. Host arm/start/finish and phase logging are diagnostic;
   device-driven admission and full backpressure remain future serving work.
3. P38 full routed compile018 covers11388 PEs with7714 ELF images and a
   maximum48112bytes including4096 stack. Reference003 verifies396953216
   original bank bytes,454400 mixed tiles,53661 auxiliary pages and every MLP
   prefix against the actual descriptors; bank hashes and layouts match P37.
   Selected simulator025 passes584 commands, all60178 original words, eight
   arrival-prepared operand cases and normal stop. All170 source/publication
   tests pass. Use the exact joint auxiliary map, including49152 GDN pages.
   The prototype mixer SDK RPC is replaced by a tagged arrival entry in routed
   builds. Device C15 events retire bounded inline batches without symbol polls;
   these improve cold control and are not a neural serving schedule.
   P39 physical002 now qualifies all five complete original projections against
   independent reference001:86400 bit-exact stored BF16 outputs, all3554 endpoints,
   zero/change/replay, full bank retention,20 drains and normal released exit.
   Connect actual conv/gates/GDN root consumers and adjacent
   stages with input and retirement credits; producer DMA completion is not
   downstream drain. Internal cohosts retain112-160bytes minimum SRAM margin;
   the actual gateway retains5840bytes in that local census. Physical P39 reaches
   the48128-byte ceiling. Additional arithmetic/state transport requires new
   placement and fresh full-program admission. Keep P31 as the accepted physical
   neural graph and preserve020-024 failures; avoid another isolated sweep.
4. P40 implements the single-conversation layer00 specialization. Compile021
   admits11388 PEs/7910 images with maximum48112bytes including4096 stack;
   reference001 verifies393746048 retained bytes, including all original weights,
   the complete first convolution/recurrent state and both work slots. Removed
   second-context pages are explicit invalid addresses. Use DialogueAuxiliaryPlacement
   and bank-specialization.json, not the two-context provenance map. The actual
   per-PE census frees3207168bytes but only59 non-gateway PEs have>=8192bytes spare.
   Use those measured locations when lowering real conv/gate consumers; account
   for distributed GDN state, original page ownership, cohost code/scratch and
   phase-specific communication resources before fresh compiler/runtime admission.
   Do not replace this with an isolated optimization sweep. Full-model context
   capacity remains historical96, not qualified multi-turn capacity.
   The new DialogueAdmission oracle covers exact-prefix appends, pending final
   token commit, reset/replay, stop/continue and overflow. Its synthetic events
   must be replaced by actual complete-stage retirement/state-clear events in
   serving integration; it is not a neural runtime or timing result.
5. Measure complete dependent stage service and optimize its critical path.
   The September28 target is one stateful multi-turn conversation at>=2000 average
   output tokens/s, not aggregate independent-request throughput. Report appended
   prefill/TTFT, pure decode/ITL and full response time. The primary count divides
   all actual assistant output tokens by summed turn response durations.
6. Instantiate embedding, all64 resident stages, the full head and token feedback.
   Preserve GDN/conv/KV state and absolute positions across new prompts. Distinguish
   emitted tokens from fully committed positions, including a pending final token.
   Freeze at least three context-dependent turns and a longer-history capacity
   workload. Qualify reference alignment, reset/replay, stop/continue, growing
   context and overflow rejection. Current context96 packing is not a multi-turn
   acceptance result; do not silently truncate or reset to fit it. See
   DIALOGUE-ACCEPTANCE.md for the active contract.

Native pair simulator003 is slower on the used shapes. Native unroll005 saves
local simulator counts but complete compile008 exceeds SRAM on68PE. Preserve
these observations and the syntax failure004; neither variant is selected for
hardware. The shared-input candidate retains the original native kernel.

Retain all qualified kernels and failed snapshots. Do not dispatch the retired
whole-wafer temporal overlay, unsupported WSE-3 color swap or simultaneous
RAMP/cardinal receive. Publish evidence and verify resource release after every
bounded experiment. No partial graph passes complete-model acceptance.
