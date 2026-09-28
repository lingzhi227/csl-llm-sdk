# Next work: resident spatial pipeline

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
   Next qualify all five complete original projections against independent
   reference001 (86400 BF16 outputs), including zero/change/replay and full bank
   retention. Then connect actual conv/gates/GDN root consumers and adjacent
   stages with input and retirement credits; producer DMA completion is not
   downstream drain. Internal cohosts retain112-160bytes minimum SRAM margin;
   the actual gateway retains5840bytes. Additional arithmetic/state transport
   requires fresh full-program admission. Keep P31 as the accepted physical
   neural graph and preserve020-024 failures; avoid another isolated sweep.
4. Measure complete dependent stage service and optimize its critical path.
   The September28 target is one stateful multi-turn conversation at>=2000 average
   output tokens/s, not aggregate independent-request throughput. Report appended
   prefill/TTFT, pure decode/ITL and full response time. The primary count divides
   all actual assistant output tokens by summed turn response durations.
5. Instantiate embedding, all64 resident stages, the full head and token feedback.
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
